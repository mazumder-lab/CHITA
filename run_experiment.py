import string
from utils.utils import *
from pruners.CHITA import CHITA
from pruners.multi_stage_pruner import MultiStagePruner
import json
import sys
import argparse
from itertools import product
import time
import os
from torch.utils.data import DataLoader
import copy



parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='mlpnet')
parser.add_argument('--dset', type=str, default='mnist')

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--exp_id',type=str,default='')

parser.add_argument('--test_batch_size', type=int, default=500)

parser.add_argument('--fisher_subsample_size', type=int, nargs='+')
parser.add_argument('--fisher_mini_bsz', type=int, nargs='+')

parser.add_argument('--num_iterations', type=int, nargs='+')
parser.add_argument('--num_stages', type=int, nargs='+')
parser.add_argument('--seed', type=int, nargs='+')
parser.add_argument('--first_order_term', type=lambda x: (str(x).lower() == 'true'), nargs='+')
parser.add_argument('--alpha_one', type=lambda x: (str(x).lower() == 'true'), nargs='+')
parser.add_argument('--sparsity', type=float, nargs='+')
parser.add_argument('--base_level', type=float, default=0.1) ##In correspondance with sparsity
parser.add_argument('--l2', type=float, nargs='+')
parser.add_argument('--l2_logspace', type=lambda x: (str(x).lower() == 'true'), default=False) ##A different way to provide l2 list by giving 3 floats
parser.add_argument('--sparsity_schedule', type=str, nargs='+')
parser.add_argument('--algo', type=str, nargs='+')
parser.add_argument('--block_size', type=int, nargs='+') ##Set to -1 if algo does not use this
parser.add_argument('--pretrained', type=lambda x: (str(x).lower() == 'true'), default=True)


args = parser.parse_args()
arch = args.arch
dset = args.dset
num_workers = args.num_workers
exp_name = args.exp_name
pretrained = args.pretrained


fisher_sizes = [(args.fisher_subsample_size[i], args.fisher_mini_bsz[i]) for i in range(len(args.fisher_mini_bsz)) ] ##Glue the sizes together so we don't take products



##Change this to path of imagenet dset
IMAGENET_PATH = os.environ['IMAGENET_PATH']
CIFAR10_PATH = '../datasets'
MNIST_PATH = '../datasets'

####
USE_DATA_PARALLEL=False

dset_paths = {'imagenet':IMAGENET_PATH+'/raw','cifar10':CIFAR10_PATH,
                'mnist':MNIST_PATH}


dset_path = dset_paths[dset]




X = None
base_level = args.base_level
ROOT_PATH = '.' ##path where results are stored


acc_different_methods = []
FOLDER = '{}/results/{}_{}_{}'.format(ROOT_PATH,arch,dset,exp_name)
FILE =  FOLDER+'/data{}_{}.csv'.format(args.exp_id,str(int(time.time())))
os.makedirs(FOLDER,exist_ok=True)

old_fisher_size,old_seed = None,None

if args.alpha_one is None:
    args.alpha_one  = [False]

if args.l2_logspace:
    if len(args.l2) != 3:
        raise ValueError('l2 arguments needs to provide exactly 3 values')
    args.l2 = np.logspace(args.l2[0],args.l2[1],int(args.l2[2]))
    print('l2 used',args.l2)

for seed,fisher_size, num_stages,num_iterations,first_order_term,sparsity,l2,sparsity_schedule,algo,block_size,alpha_one in product(args.seed, fisher_sizes,args.num_stages, args.num_iterations,args.first_order_term,args.sparsity,args.l2,args.sparsity_schedule,args.algo,args.block_size,args.alpha_one):
    
    print('seed,fisher_size, num_stages,num_iterations,first_order_term,sparsity,l2,sparsity_schedule,algo,block_size',seed,fisher_size, num_stages,num_iterations,first_order_term,sparsity,l2,sparsity_schedule,block_size)

    
    if (algo != 'Heuristic_LSBlock') and  (block_size != -1): ##Only Heuristic_LSBlock uses a block_dize
        continue

    if (algo == 'Heuristic_LSBlock') and  (block_size == -1):
        continue

    if  seed != old_seed or old_fisher_size != fisher_size:
        X = None


    if seed != old_seed:
        set_seed(seed)

        model,train_dataset,test_dataset,criterion,modules_to_prune = model_factory(arch,dset_path,pretrained=pretrained)


        train_dataloader = DataLoader(train_dataset, batch_size=args.test_batch_size, shuffle=True,num_workers=num_workers,pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True,num_workers=num_workers,pin_memory=True)

        ##########
        recompute_bn_stats = (arch == 'resnet20') #Resnet20 models need updating their BN statistics
        if recompute_bn_stats:
            device='cpu'
            model.train()
            original_acc = compute_acc(model,train_dataloader,device=device) #sets up bn stats
            model.eval()
        ################
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device: ', device)

        if torch.cuda.device_count()  > 1 and USE_DATA_PARALLEL:
            print('Using DataParallel with',torch.cuda.device_count(),'GPUs')
            model = torch.nn.DataParallel(model)
            modules_to_prune = ["module." + x for x in modules_to_prune]

        model.to(device)
        model.eval()
        start_test = time.time()
        dense_acc = compute_acc(model,test_dataloader,device=device)
        time_test = time.time() - start_test
        print('Dense test accuracy', dense_acc,' computation time : ', time_test)
        ###############


    fisher_subsample_size, fisher_mini_bsz = fisher_size

    prun_dataloader = DataLoader(train_dataset, batch_size=fisher_mini_bsz, shuffle=True,num_workers=num_workers,pin_memory=True)

    model_pruned = copy.deepcopy(model)
    pruner = CHITA(model_pruned,modules_to_prune,prun_dataloader,fisher_subsample_size,fisher_mini_bsz,criterion,block_size,l2,num_iterations,
    first_order_term,alpha_one,
    device,algo)
    mask = torch.ones_like(get_pvec(model,modules_to_prune)).cpu() != 0
    multi_stage_pruner = MultiStagePruner(pruner,test_dataloader,sparsity_schedule,num_stages)

    start = time.time()
    w_pruned,mask = multi_stage_pruner.prune(mask,sparsity,base_level,grads=X)
    end=time.time()
    del model_pruned
    acc_different_methods.append({'algo':algo,'sparsity':sparsity,'l2':l2,
    'first_order_term':first_order_term,'runtime':end-start,'seed':seed,'fisher_mini_bsz':fisher_mini_bsz,'fisher_subsample_size':fisher_subsample_size,
    'num_iterations':num_iterations,'num_stages':num_stages,'recompute_bn_stats' : recompute_bn_stats,'ignore_bias':True,
    'base_level':base_level,'sparsity_schedule':sparsity_schedule,'block_size':block_size,'alpha_one':alpha_one})
    acc_different_methods[-1]['results'] = multi_stage_pruner.results
    with open(FILE, "w") as file:
        json.dump(acc_different_methods, file,cls=NpEncoder)

    old_fisher_size,old_seed = fisher_size,seed
    if X is None:
        X=multi_stage_pruner.pruner.grads


