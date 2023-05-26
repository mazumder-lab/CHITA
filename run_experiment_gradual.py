import string
from utils.main_utils import *
from pruners.CHITA import CHITA
from pruners.multi_stage_pruner import MultiStagePruner
from pruners.gradual_pruner import GradualPruner
import json
import sys
import argparse
from itertools import product
import time
import os
from torch.utils.data import DataLoader
import copy
from utils.lr_schedules import cosine_lr_restarts,mfac_lr_schedule
import torch.distributed as dist
import builtins


parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='mlpnet')
parser.add_argument('--dset', type=str, default='mnist')

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--exp_id',type=str,default='')

parser.add_argument('--train_batch_size', type=int, default=500)
parser.add_argument('--test_batch_size', type=int, default=500)

parser.add_argument('--fisher_subsample_size', type=int, nargs='+')
parser.add_argument('--fisher_mini_bsz', type=int, nargs='+')

parser.add_argument('--num_iterations', type=int, nargs='+')
parser.add_argument('--num_stages', type=int, nargs='+')
parser.add_argument('--seed', type=int, nargs='+')
parser.add_argument('--first_order_term', type=lambda x: (str(x).lower() == 'true'), nargs='+')
parser.add_argument('--sparsity', type=float, nargs='+')
parser.add_argument('--base_level', type=float, default=0.1) ##In correspondance with sparsity
parser.add_argument('--outer_base_level',type=float,default=0.5)
parser.add_argument('--l2', type=float, nargs='+')
parser.add_argument('--sparsity_schedule', type=str, nargs='+')
parser.add_argument('--algo', type=str, nargs='+')
parser.add_argument('--block_size', type=int, nargs='+') ##Set to -1 if algo does not use this

parser.add_argument('--weight_decay',type=float,default=0.00003751757813)
parser.add_argument('--momentum',type=float,default=0.9)
parser.add_argument('--max_lr',type=float)
parser.add_argument('--min_lr',type=float)
parser.add_argument('--ft_min_lr',type=float,default=-1)
parser.add_argument('--ft_max_lr',type=float,default=-1)
parser.add_argument('--prune_every',type=int)
parser.add_argument('--nprune_epochs',type=int)
parser.add_argument('--nepochs',type=int)
parser.add_argument('--gamma_ft',type=float,default=-1)
parser.add_argument('--warm_up', type=int, default=0)
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--first_epoch', type=int, default=0)
parser.add_argument('--schedule', type=str, default='cosine_lr_restarts')
parser.add_argument('--pretrained', type=lambda x: (str(x).lower() == 'true'), default=True)

parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')

args = parser.parse_args()
arch = args.arch
dset = args.dset
num_workers = args.num_workers
exp_name = args.exp_name
pretrained = args.pretrained
momentum = args.momentum
weight_decay = args.weight_decay


fisher_sizes = [(args.fisher_subsample_size[i], args.fisher_mini_bsz[i]) for i in range(len(args.fisher_mini_bsz)) ] ##Glue the sizes together so we don't take products

if "WORLD_SIZE" in os.environ:
    args.world_size = int(os.environ["WORLD_SIZE"])
else:
    args.world_size = 1
args.distributed = args.world_size > 1

if args.distributed:
    if 'SLURM_PROCID' in os.environ: # for slurm scheduler
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif args.local_rank != -1: # for torch.distributed.launch
        args.rank = args.local_rank
        args.gpu = args.local_rank
else:
    args.rank = 0
    args.gpu=0




##Change this to path of imagenet dset
if 'IMAGENET_PATH' in os.environ:  
    IMAGENET_PATH = os.environ['IMAGENET_PATH']
else:
    print('****Warning**** No IMAGENET_PATH variable')
    IMAGENET_PATH = ''
CIFAR10_PATH = '../datasets'
MNIST_PATH = '../datasets'

dset_paths = {'imagenet':IMAGENET_PATH,'cifar10':CIFAR10_PATH,
                'mnist':MNIST_PATH}

dset_path = dset_paths[dset]

model,train_dataset,test_dataset,criterion,modules_to_prune = model_factory(arch,dset_path,pretrained=pretrained)


###### Load from checkpoint
if len(args.checkpoint_path) >0:
    checkpoint = torch.load(args.checkpoint_path)
    new_state_trained = OrderedDict()
    for k in checkpoint['model_state_dict']:
        new_state_trained[k[7:]] = checkpoint['model_state_dict'][k]
    model.load_state_dict(new_state_trained)


####
if args.distributed:
    device = args.gpu
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device ', device)


ngpus_per_node = torch.cuda.device_count()

if args.distributed:
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.world_size, rank=args.rank)
    torch.backends.cudnn.benchmark = True

    if args.rank!=0:
        def print_pass(*args):
            pass
        #builtins.print = print_pass ##Uncomment to block all other ranks from printing

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
    #modules_to_prune = ["module." + x for x in modules_to_prune]
    modules_to_prune = [x for x in modules_to_prune]
else:
    model = model.to(device)
    model_without_ddp = model

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
else:
    train_sampler = None
train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=(train_sampler is None),num_workers=num_workers,pin_memory=True,sampler=train_sampler)
test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True,num_workers=num_workers,pin_memory=True)

alpha_one=False

##########
recompute_bn_stats = (arch == 'resnet20')
if recompute_bn_stats and pretrained:
    #device='cpu'
    model.train()
    original_acc = compute_acc(model,train_dataloader,device=device) #sets up bn stats
    model.eval()
################
#model.to(device)
model.eval()
if (not args.distributed or  args.rank == 0):
    start_test = time.time()
    dense_acc = compute_acc(model_without_ddp,test_dataloader,device=device)
    time_test = time.time() - start_test
    print('Dense test accuracy', dense_acc,' computation time : ', time_test)
###############

acc_different_methods = []
X = None
base_level = args.base_level
ROOT_PATH = '.'


acc_different_methods = []
FOLDER = '{}/results/{}_{}_{}'.format(ROOT_PATH,arch,dset,exp_name)
FILE =  FOLDER+'/data{}_{}.csv'.format(args.exp_id,str(int(time.time())))
os.makedirs(FOLDER,exist_ok=True)

old_fisher_size,old_seed = None,None

torch.backends.cudnn.benchmark = True

for seed,fisher_size, num_stages,num_iterations,first_order_term,sparsity,l2,sparsity_schedule,algo,block_size in product(args.seed, fisher_sizes,args.num_stages,args.num_iterations,args.first_order_term,args.sparsity,args.l2,args.sparsity_schedule,args.algo,args.block_size):
    
    

    if (algo == 'Heuristic_LSBlock') and  (block_size == -1):
        continue


    
    print('seed,fisher_size, num_stages,num_iterations,first_order_term,sparsity,l2,sparsity_schedule,algo,block_size',seed,fisher_size, num_stages,num_iterations,first_order_term,sparsity,l2,sparsity_schedule,algo,block_size)

    X = None
    nepochs = args.nepochs
    nprun_epochs = args.nprune_epochs
    reset_optimizer=True
    sparsities = generate_schedule(nprun_epochs,args.outer_base_level,sparsity,'poly')
    prun_every = args.prune_every
    gamma_ft = args.gamma_ft
    prunepochs = [(i) *prun_every for i in range(len(sparsities))]
    ignore_bias = True

    if args.ft_max_lr == -1:
        args.ft_max_lr = None
    if args.ft_min_lr == -1:
        args.ft_min_lr = None

    if args.schedule == 'mfac':
        lr_schedule = mfac_lr_schedule(nepochs,args.max_lr,args.min_lr,nprun_epochs,args.warm_up)
        print(lr_schedule)
    elif args.schedule == 'cosine_lr_restarts':
        lr_schedule = cosine_lr_restarts(nepochs,args.max_lr,args.min_lr,nprun_epochs,prun_every,gamma_ft,args.warm_up,ft_max_lr=args.ft_max_lr,ft_min_lr=args.ft_min_lr)
        print(args.ft_max_lr,args.ft_min_lr,'----<>')
    else:
        print('Unrecognized schedule')
        break

    fisher_subsample_size, fisher_mini_bsz = fisher_size
    acc_different_methods.append({'algo':algo,'l2':l2,'max_lr':args.max_lr,'min_lr':args.min_lr,'nepochs':args.nepochs,
    'nprun_epochs':nprun_epochs,'prun_every':prun_every,
    'first_order_term':first_order_term,'seed':seed,'fisher_mini_bsz':fisher_mini_bsz,
    'fisher_subsample_size':fisher_subsample_size,
    'num_iterations':num_iterations,'num_stages':num_stages,'recompute_bn_stats' : recompute_bn_stats,
    'ignore_bias':True,'base_level':base_level,
    'sparsity_schedule':sparsity_schedule,
    'block_size':block_size,'train_batch_size':args.train_batch_size,
    'test_batch_size':args.test_batch_size})


    prun_dataloader = DataLoader(train_dataset, batch_size=fisher_mini_bsz, shuffle=True,num_workers=num_workers,pin_memory=True)

    
    model_pruned = model_without_ddp
    pruner = CHITA(model_pruned,modules_to_prune,prun_dataloader,fisher_subsample_size,fisher_mini_bsz,criterion,block_size,l2,num_iterations,
    first_order_term,alpha_one,device,algo)

    multi_stage_pruner = MultiStagePruner(pruner,test_dataloader,sparsity_schedule,num_stages)
    mask = get_pvec(model_without_ddp,modules_to_prune).cpu() != 0
    gradual_pruner = GradualPruner(multi_stage_pruner,train_dataloader,test_dataloader,criterion,
        modules_to_prune,reset_optimizer,momentum,weight_decay,acc_different_methods,FILE,seed,model=model,device=device,mask=mask,
        first_epoch=args.first_epoch,distributed=args.distributed,rank=args.rank,world_size=args.world_size)

    gradual_pruner.prune(nepochs,lr_schedule,prunepochs,sparsities)

    with open(FILE, "w") as file:
        json.dump(acc_different_methods, file,cls=NpEncoder)

    old_fisher_size,old_seed = fisher_size,seed


