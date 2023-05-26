

TASK_ID=0
EXP_ID="0_0"

CHECKPOINT_PATH=""
FIRST_EPOCH=0

echo $TASK_ID
echo $EXP_ID

algos=("Heuristic_LSBlock" "MP")
block_sizes=(500 -1)
split_types=(1 -1)
algo=${algos[0]}
block_size=${block_sizes[0]}
split_type=${split_types[0]}

nums_stages=(1 16 16)



sparsity_schedule="poly"

training_schedules=("cosine_fast_works_098" "cosine_fast1" "cosine_one")
training_schedule=${training_schedules[TASK_ID%3]}
num_stages=${nums_stages[TASK_ID%3]}

if [ $training_schedule == "cosine_fast" ] 
then 
    max_lr=0.05
    min_lr=0.000005
    prune_every=15
    nprune_epochs=7
    nepochs=100
    warm_up=0
    ft_max_lr=0.0005
    ft_min_lr=0.00005
    gamma_ft=0.5
fi
if [ $training_schedule == "cosine_fast_works_098" ] 
then 
    max_lr=0.1
    min_lr=0.00001
    prune_every=12
    nprune_epochs=7
    nepochs=100
    warm_up=0
    ft_max_lr=0.1
    ft_min_lr=0.00001
    gamma_ft=-1
fi
if [ $training_schedule == "cosine_fast1" ] 
then 
    max_lr=0.05
    min_lr=0.000005
    prune_every=12
    nprune_epochs=7
    nepochs=100
    warm_up=0
    gamma_ft=-1
fi
if [ $training_schedule == "cosine_fast_works" ] 
then 
    max_lr=0.05
    min_lr=0.000005
    prune_every=15
    nprune_epochs=7
    nepochs=100
    warm_up=0
    gamma_ft=-1
fi
if [ $training_schedule == "cosine_fast_gamma" ] 
then 
    max_lr=0.05
    min_lr=0.000005
    prune_every=15
    nprune_epochs=7
    nepochs=150
    warm_up=0
    gamma_ft=0.8
fi
if [ $training_schedule == "cosine_one" ] 
then 
    max_lr=0.256
    min_lr=0.000005
    prune_every=1
    nprune_epochs=1
    nepochs=100
    warm_up=5
    gamma_ft=-1
fi
if [ $training_schedule == "cosine_slow" ]
then
    max_lr=0.005
    min_lr=0.000005
    prune_every=4
    nprune_epochs=16
    nepochs=100
    warm_up=0
    gamma_ft=-1
fi
if [ $training_schedule == "cosine_fast_slr" ]
then
    max_lr=0.005
    min_lr=0.000005
    prune_every=12
    nprune_epochs=7
    nepochs=100
    warm_up=0
    gamma_ft=0.9
fi

echo $max_lr

seed=2

fisher_subsample_sizes=(500)
fisher_subsample_size=${fisher_subsample_sizes[0]}

l2s=(0.0001 0.001)
l2=${l2s[0]}

fisher_mini_bszs=(1)
fisher_mini_bsz=16

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$((12073 + TASK_ID))
export WORLD_SIZE=1
echo $MASTER_PORT

#export MASTER_ADDR=$master_addr


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 run_experiment_gradual.py --arch resnet20 --dset cifar10 --num_workers 2 \
--exp_name test --exp_id ${EXP_ID} --test_batch_size 256 --train_batch_size 256 \
--fisher_subsample_size ${fisher_subsample_size} --fisher_mini_bsz ${fisher_mini_bsz} \
--num_iterations 1 --num_stages ${num_stages} --seed ${seed} \
--first_order_term False --sparsity 0.9 --base_level 0.3 \
--outer_base_level 0.5  --l2 ${l2} --sparsity_schedule ${sparsity_schedule} \
--algo ${algo} --block_size ${block_size} \
--max_lr ${max_lr} --min_lr ${min_lr} --prune_every ${prune_every} --nprune_epochs ${nprune_epochs} \
--nepochs ${nepochs} --gamma_ft ${gamma_ft} --warm_up ${warm_up} --ft_max_lr ${ft_max_lr} --ft_min_lr ${ft_min_lr}