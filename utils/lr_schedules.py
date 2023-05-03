import numpy as np

def cosine_lr(epochs,max_lr,min_lr):
    return [min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * epoch / epochs)) for epoch in range(epochs)]


def cosine_lr_restarts(epochs,max_lr,min_lr,n_prune_epochs,prune_every,gamma_ft=-1,warm_up=0,ft_max_lr=None,ft_min_lr=None):
    lr_list =[max_lr]*warm_up
    epochs -= warm_up
    for _ in range(n_prune_epochs-1):
        lr_list += cosine_lr(prune_every,max_lr,min_lr)
    if ft_min_lr is None:
        ft_min_lr = min_lr
    if ft_max_lr is None:
        ft_max_lr = max_lr
    if gamma_ft < 0:
        lr_list += cosine_lr(epochs - prune_every*(n_prune_epochs-1),ft_max_lr,ft_min_lr)
    else:
        lr_list += [ft_max_lr*np.power(gamma_ft,i)  for i in range(epochs - prune_every*(n_prune_epochs-1)) ]
    return lr_list

def mfac_lr_schedule(epochs,max_lr,min_lr,n_prune_epochs,warm_up=0):
    lr_list =[max_lr]*warm_up
    epochs -= warm_up
    for _ in range(n_prune_epochs-1):
        lr_list += [max_lr,max_lr,min_lr]
    
    lr_list += [max_lr]*(74 - len(lr_list))
    max_lr /= 10
    lr_list += [max_lr]*(89 - len(lr_list))
    max_lr /= 10
    lr_list += [max_lr]*(epochs - len(lr_list))
    return lr_list    


def weight_rewinding_mobilenetv1(epochs):
    warmup = 5
    max_lr = 0.256
    min_lr = 0
    lr_list = [max_lr]*warmup + cosine_lr_restarts(epochs - warmup,max_lr,min_lr,1,epochs - warmup)
    momentum = 0.875
    weight_decay = 0.00003751757813
    label_smoothing: 0.1
