

TASK_ID=0
EXP_ID="0_0"

CHECKPOINT_PATH="/home/rbenbaki/CHITA/results/resnet50_imagenet_0.9_seed_2/data2_1683490689.csv_epoch90.pth"
FIRST_EPOCH=91

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


if [ $training_schedule == "cosine_fast_works_098" ] 
then 
    max_lr=0.1
    min_lr=0.00001
    prune_every=12
    nprune_epochs=7
    nepochs=100
    warm_up=0
    ft_max_lr=0.05
    ft_max_lr=0.1
    ft_min_lr=0.00001
    gamma_ft=-1
fi

echo $max_lr


fisher_subsample_sizes=(500)
fisher_subsample_size=${fisher_subsample_sizes[0]}

l2s=(0.0001 0.001)
l2=${l2s[0]}

fisher_mini_bszs=(1)
fisher_mini_bsz=16

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$((12073 + TASK_ID))
echo $MASTER_PORT

#export MASTER_ADDR=$master_addr

export OMP_NUM_THREADS=24


CHECKPOINT_PATH="/home/rbenbaki/CHITA/results/resnet50_imagenet_0.9_seed_2/data2_1683490689.csv_epoch71.pth"
FIRST_EPOCH=72
seed=2
sparsity=0.9
EXP_NAME="${sparsity}_seed_${seed}_FT0.1"
echo $EXP_NAME

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 run_experiment_gradual.py --arch resnet50 --dset imagenet --num_workers 12 \
--exp_name ${EXP_NAME} --exp_id ${seed} --test_batch_size 256 --train_batch_size 256 \
--fisher_subsample_size ${fisher_subsample_size} --fisher_mini_bsz ${fisher_mini_bsz} \
--num_iterations 1 --num_stages ${num_stages} --seed ${seed} \
--first_order_term False --sparsity ${sparsity} --base_level 0.3 \
--outer_base_level 0.5  --l2 ${l2} --sparsity_schedule ${sparsity_schedule} \
--algo ${algo} --block_size ${block_size} \
--max_lr ${max_lr} --min_lr ${min_lr} --prune_every ${prune_every} --nprune_epochs ${nprune_epochs} \
--nepochs ${nepochs} --gamma_ft ${gamma_ft} --warm_up ${warm_up} --ft_max_lr ${ft_max_lr} --ft_min_lr ${ft_min_lr} \
--first_epoch ${FIRST_EPOCH} --checkpoint_path ${CHECKPOINT_PATH}