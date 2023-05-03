from ..utils.utils import *
import time
import torch.distributed as dist
import gc
import autograd_hacks.autograd_hacks as autograd_hacks


class GradualPruner:
    def __init__(self,multi_stage_pruner,train_dataloader,test_dataloader,criterion,
        params,reset_optimizer,momentum,weight_decay,results,filename,seed,mask=None,model=None,device=None,first_epoch=0,distributed=False,rank=-1,world_size=1):
        set_seed(seed)
        assert not distributed or rank >=0
        self.pruner=multi_stage_pruner       #Pruner may have a subobject of model 
        self.model=model
        self.device=device
        self.train_dataloader=train_dataloader
        self.test_dataloader=test_dataloader
        self.criterion=criterion
        self.params=params
        self.reset_optimizer=reset_optimizer
        self.momentum=momentum
        self.weight_decay=weight_decay
        self.results=results #List of dictionnaries holding the results
        self.filename=filename #Filename to write results
        if mask is None:
            self.mask=torch.ones_like(get_pvec(self.model_without_ddp,self.params)).cpu() != 0
        else:
            self.mask = mask
        self.mask = self.mask.to(self.device)
        self.optim=torch.optim.SGD(self.model.parameters(), lr=0, momentum=self.momentum, weight_decay=self.weight_decay)
        self.runloss=0
        self.step=0
        self.first_epoch=first_epoch
        self.distributed=distributed
        self.rank = rank
        self.world_size=world_size
        if self.distributed:
            self.model_without_ddp = self.pruner.pruner.model
        else:
            self.model_without_ddp = self.model
        print('Initial Model has sparsity',(~self.mask.cpu().numpy()).mean(), ' at rank',self.rank)
        

    def train(self):
        torch.cuda.empty_cache()
        gc.collect()
        for x, y in self.train_dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            zero_grads(self.model)
            loss = self.criterion(self.model(x), y)
            #self.runloss = .99 * self.runloss + .01 * loss.item()
            loss.backward()
            self.optim.step()
            apply_mask(self.mask,self.model_without_ddp,self.params,self.device)
            self.step+=1
            if self.step % 100 == 0:
                print('step %06d: runloss=%.3f' % (self.step, self.runloss), self.rank)
        torch.cuda.empty_cache()
        gc.collect()

    def prune(self,nepochs,lr_schedule,prunepochs,sparsities,base_level_=0.1):

        for key in ['epoch','pruning_res','running_loss','acc','lr','momentum','weight_decay']:
            self.results[-1][key] = []

        for epoch in range(self.first_epoch,nepochs):
            if (self.distributed):
                self.train_dataloader.sampler.set_epoch(epoch)
            if epoch in prunepochs and (not self.distributed or self.rank == 0):
                if self.reset_optimizer: ##Reset momentum
                    self.optim = torch.optim.SGD(self.model.parameters(), lr=0, momentum=self.momentum, weight_decay=self.weight_decay)
                epoch_index = prunepochs.index(epoch)
                if epoch_index > 0:
                    base_level = sparsities[epoch_index-1]
                else:
                    base_level = base_level_
                self.model.eval()
                autograd_hacks.enable_hooks()
                w_pruned,mask=self.pruner.prune(self.mask,sparsities[epoch_index],base_level)
                autograd_hacks.disable_hooks()
                self.mask = mask.to(self.device)
                del mask
                del w_pruned
                pruning_res = self.pruner.results
                self.pruner.reset_pruner()
                self.model.train()
            else:
                pruning_res=[]
            
            if self.distributed:
                dist.barrier()
                sync_weights(self.model,self.rank,self.world_size)
                sync_mask(self,self.rank,self.world_size)
                print('Done syncing at',self.rank, 'sparsity',(~(get_pvec(self.model.module,self.params).cpu() != 0).numpy()).mean(),'mask',(~self.mask.cpu().numpy()).mean())
            ##Set learning rate
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr_schedule[epoch]
            
            print('starting epoch',epoch, '-- lr',lr_schedule[epoch],' at rank',self.rank)

            self.model.train()
            start_epoch = time.time()
            self.train()
            end_epoch = time.time()
            self.model.eval()


            print('epoch ',epoch, ' - time :',end_epoch-start_epoch,' at rank',self.rank)
            if self.distributed:
                dist.barrier()
            if (not self.distributed or self.rank == 0):
                acc = compute_acc(self.model_without_ddp,self.test_dataloader,self.device)
                print('epoch ',epoch, ' - acc :',acc, ' - time :',end_epoch-start_epoch,' at rank',self.rank)
                PATH = self.filename+'_epoch'+str(epoch)+'.pth'
                torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict()}, PATH)
                self.results[-1]['epoch'].append(epoch)
                self.results[-1]['pruning_res'].append(pruning_res)
                self.results[-1]['running_loss'].append(self.runloss)
                self.results[-1]['acc'].append(acc)
                self.results[-1]['lr'].append(lr_schedule[epoch])
                self.results[-1]['momentum'].append(self.momentum)
                self.results[-1]['weight_decay'].append(self.weight_decay)

                with open(self.filename, "w") as file:
                    json.dump(self.results, file,cls=NpEncoder)