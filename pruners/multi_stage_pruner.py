import re
import time
from utils import *


class MultiStagePruner:
    def __init__(self,pruner,test_dataloader,schedule,num_stages):
        self.pruner = pruner
        self.test_dataloader = test_dataloader
        self.results = []
        self.schedule=schedule
        self.num_stages=num_stages

    def reset_pruner(self):
        self.results=[]

    def get_input_dim(self):
        for i,o in self.test_dataloader:
            break
        return list(i.shape[1:])
    
    def prune(self,mask0,sparsity,base_level,grads=None,FILE=None):
        sparsities=generate_schedule(self.num_stages, base_level,sparsity,self.schedule)
        mask = torch.clone(mask0)
        input_dim = self.get_input_dim()
        if self.num_stages > 1:
            grads=None
        for i,sparsity_stg in enumerate(sparsities):
            self.pruner.reset_pruner()
            start = time.time()
            w_pruned,mask = self.pruner.prune(mask,sparsity_stg,grads=grads)
            end = time.time()
            print('Stage took', end-start)
            self.pruner.update_model(w_pruned)
            self.pruner.compute_flops(input_dim)
            self.results.append(self.pruner.results)
            if  not self.test_dataloader is None:
                self.pruner.model.eval()
                self.results[-1]['sparsity'] = sparsity_stg
                self.results[-1]['test_acc'] = compute_acc(self.pruner.model,self.test_dataloader,self.pruner.device)
                print('Stage sp',sparsity_stg,self.results[-1]['test_acc'])
                if not FILE is None:
                    with open(FILE+'_stage'+str(i), "w") as file:
                        json.dump(self.results, file,cls=NpEncoder)
        return w_pruned,mask