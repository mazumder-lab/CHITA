import torch
import sys
import numpy as np
import os
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.models import resnet50 as torch_resnet50
from models.resnet_cifar10 import resnet20
from models.wideresnet_cifar import Wide_ResNet
from models.mlpnet import MlpNet
from models.mobilenet import mobilenet
from collections import OrderedDict
import json
import torch.distributed as dist


from CHITA_opt.L0_card_const import Heuristic_CD_PP,Active_IHTCDLS_PP,Heuristic_LS,Heuristic_LSBlock,evaluate_obj

def sync_weights(model, rank, world_size):
    for param in model.parameters():
        if rank == 0:
            # Rank 0 is sending it's own weight
            # to all it's siblings (1 to world_size)
            for sibling in range(1, world_size):
                dist.send(param.data, dst=sibling)
        else:
            # Siblings must recieve the parameters
            dist.recv(param.data, src=0)

def sync_mask(pruner, rank, world_size):
    if rank == 0:
        # Rank 0 is sending it's own weight
        # to all it's siblings (1 to world_size)
        for sibling in range(1, world_size):
            dist.send(pruner.mask.data, dst=sibling)
    else:
        # Siblings must recieve the parameters
        dist.recv(pruner.mask.data, src=0)

def flatten_tensor_list(tensors):
    flattened = []
    for tensor in tensors:
        flattened.append(tensor.view(-1))
    return torch.cat(flattened, 0)


def print_parameters(model):
    for name, param in model.named_parameters(): 
        print(name, param.shape)

def load_model(path, model):
    tmp = torch.load(path, map_location='cpu')
    if 'state_dict' in tmp:
        tmp = tmp['state_dict']
    if 'model' in tmp:
        tmp = tmp['model']
    for k in list(tmp.keys()):
        if 'module.' in k:
            tmp[k.replace('module.', '')] = tmp[k]
            del tmp[k]
    model.load_state_dict(tmp)


def imagenet_get_datasets(data_dir):

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    train_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()
    ]
    
    train_transform += [
        transforms.ToTensor(),
        normalize,
    ]
    train_transform = transforms.Compose(train_transform)

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset


@torch.no_grad()
def get_pvec(model, params):
    state_dict = model.state_dict()
    return torch.cat([
        state_dict[p].reshape(-1) for p in params
    ])

@torch.no_grad()
def get_sparsity(model, params):
    pvec = get_pvec(model,params)
    return (pvec == 0).float().mean()

@torch.no_grad()
def get_blocklist(model,params,block_size):
    i_w = 0
    block_list = [0]
    state_dict = model.state_dict()
    for p in params:
        param_size = np.prod(state_dict[p].shape)
        if param_size <block_size:
            block_list.append(i_w+param_size)
        else:
            num_block = int(param_size/block_size)
            block_subdiag = list(range(i_w,i_w+param_size+1,int(param_size/num_block))) 
            block_subdiag[-1] = i_w+param_size
            block_list += block_subdiag   
        i_w += param_size
    return block_list

@torch.no_grad()
def set_pvec(w, model, params,device, nhwc=False):
    state_dict = model.state_dict()
    i = 0
    for p in params:
        count = state_dict[p].numel()
        if type(w) ==  torch.Tensor :
            state_dict[p] = w[i:(i + count)].reshape(state_dict[p].shape)
        else:
            state_dict[p] = torch.Tensor(w[i:(i + count)]).to(device).reshape(state_dict[p].shape)
        i += count
    model.load_state_dict(state_dict)

@torch.no_grad()
def get_gvec(model, params):
    named_parameters = dict(model.named_parameters())
    return torch.cat([
        named_parameters[p].grad.reshape(-1) for p in params
    ])
@torch.no_grad()
def get_gvec1(model, params):
    named_parameters = dict(model.named_parameters())
    return torch.cat([
        named_parameters[p].grad_sample.reshape(named_parameters[p].grad_sample.shape[0],-1) for p in params
    ],dim=1)

@torch.no_grad()
def get_gps_vec(model, params):
    named_parameters = dict(model.named_parameters())
    return torch.cat([
        named_parameters[p].grad_sample.reshape(named_parameters[p].grad_sample.shape[0],-1) for p in params
    ],dim=1)
@torch.no_grad()
def apply_mask(mask, model, params,device):
    state_dict = model.state_dict()
    i = 0
    for p in params:
        param = state_dict[p]
        count = param.numel()
        state_dict[p] *= mask[i:(i + count)].to(device).reshape(param.shape).float()
        i += count
    model.load_state_dict(state_dict)
    
@torch.no_grad()
def zero_grads(model):
    for p in model.parameters():
        p.grad = None

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_acc(model,dataloader,device='cpu',verbose=False):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    i = 0
    with torch.no_grad():
        for data in dataloader:
            i+=1
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images=images
            labels=labels
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if verbose and i%10 == 0:
                print(total,correct)

            del images,labels,outputs

    return 100 * correct / total

def compute_loss(model,criterion,dataloader,device='cpu',verbose=False):
    avg_loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    i = 0
    with torch.no_grad():
        for data in dataloader:
            i+=1
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images=images
            labels=labels
            # calculate outputs by running images through the network
            outputs = model(images)
            loss = criterion(outputs, labels).item()
            avg_loss += loss
            if verbose and i%100 ==0:
                print('computing loss', i)

            del images,labels,outputs

    return avg_loss / i


def generate_schedule(num_stages, base_level,sparsity_level,schedule):
    repeat=1
    if num_stages == 1:
        return [sparsity_level]
    if schedule == 'exp':
        sparsity_multiplier = (sparsity_level - base_level)*np.power(2, num_stages-1)/(np.power(2, num_stages-1) - 1)
        l =[base_level + sparsity_multiplier*((np.power(2, stage) - 1)/np.power(2, stage)) for stage in range(num_stages)]
        return [x for x in l for _ in range(repeat)]
    elif schedule == 'poly':
        l= [sparsity_level + (base_level-sparsity_level)*np.power(1 - (stage/(num_stages-1)), 3) for stage in range(num_stages)]
        return [x for x in l for _ in range(repeat)]
    elif schedule == 'const':
        return [sparsity_level for stage in range(num_stages)]
    elif schedule == 'linear':
        return [base_level + stage*(sparsity_level - base_level)/(num_stages-1) for stage in range(num_stages)]
    elif schedule == 'MFAC':
        sparsity_multiplier = ((1. - sparsity_level) / (1. - base_level)) ** (1./num_stages)
        return [1. - ((1. - base_level) * (sparsity_multiplier**(stage+1))) for stage in range(num_stages)]

def mnist_get_datasets(data_dir):
    # same used in hessian repo!
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=data_dir, train=True,
                                   download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root=data_dir, train=False,
                                  transform=test_transform)

    return train_dataset, test_dataset

def model_factory(arch,dset_path,pretrained=True):
    if arch == 'mlpnet':
        model = MlpNet(args=None,dataset='mnist')
        train_dataset,test_dataset = mnist_get_datasets(dset_path)
        criterion = torch.nn.functional.nll_loss

        state_trained = torch.load('checkpoints/mnist_25_epoch_93.97.ckpt',map_location=torch.device('cpu'))['model_state_dict']
        new_state_trained = OrderedDict()
        for k in state_trained:
            if 'mask' in k:
                continue
            new_state_trained[k.split('.')[1]+'.'+k.split('.')[3]] = state_trained[k]
        if pretrained:
            model.load_state_dict(new_state_trained)

        modules_to_prune = []
        for name, param in model.named_parameters():
            #print("name is {} and shape of param is {} \n".format(name, param.shape))
            layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
            if param_name == 'bias':
                    continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)
        return model,train_dataset,test_dataset,criterion,modules_to_prune
    elif arch == 'resnet20':
        state_trained = torch.load('checkpoints/resnet20_cifar10.pth.tar',map_location=torch.device('cpu'))['state_dict']
        new_state_trained = OrderedDict()
        for k in state_trained:
            new_state_trained[k[7:]] = state_trained[k]

        model = resnet20()
        if pretrained:
            model.load_state_dict(new_state_trained)

        test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_random_transforms=True

        if train_random_transforms:
            train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        train_dataset = datasets.CIFAR10(root=dset_path, train=True, download=True,transform=train_transform)
        test_dataset = datasets.CIFAR10(root=dset_path, train=False, download=True,transform=test_transform)

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, param in model.named_parameters():
            #print("name is {} and shape of param is {} \n".format(name, param.shape))
            layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
            if param_name == 'bias':
                    continue
            if 'conv' in layer_name or 'fc' in layer_name:
                modules_to_prune.append(name)

        return model,train_dataset,test_dataset,criterion,modules_to_prune
    elif arch == 'mobilenetv1':
        model = mobilenet()
        train_dataset,test_dataset = imagenet_get_datasets(dset_path)

        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                modules_to_prune.append(name+'.weight')


        if pretrained:
            path = 'checkpoints/MobileNetV1-Dense-STR.pth'
            state_trained = torch.load(path,map_location=torch.device('cpu'))['state_dict']
            new_state_trained = model.state_dict()
            for k in state_trained:
                key = k[7:]
                if key in new_state_trained:
                    new_state_trained[key] = state_trained[k].view(new_state_trained[key].size())
                else:
                    print('Missing key',key)
            model.load_state_dict(new_state_trained,strict=False)

        return model,train_dataset,test_dataset,criterion,modules_to_prune

    elif arch == 'resnet50':
        model = torch_resnet50(weights=None)
        train_dataset,test_dataset = imagenet_get_datasets(dset_path)
        criterion = torch.nn.functional.cross_entropy

        modules_to_prune = []
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                modules_to_prune.append(name+'.weight')
        print('Pruning modeules',modules_to_prune)
        if pretrained:
            
            path = 'checkpoints/ResNet50-Dense.pth'
            #path = 'checkpoints/resnet50-19c8e357.pth'
            
            state_trained = torch.load(path,map_location=torch.device('cpu'))['state_dict']
            new_state_trained = model.state_dict()
            for k in state_trained:
                key = k[7:]
                if key in new_state_trained:
                    new_state_trained[key] = state_trained[k].view(new_state_trained[key].size())
                else:
                    print('Missing key',key)
            model.load_state_dict(new_state_trained,strict=False)
            
            #model.load_state_dict(torch.load(path))

        return model,train_dataset,test_dataset,criterion,modules_to_prune



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)