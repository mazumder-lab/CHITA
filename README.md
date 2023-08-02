# Fast as CHITA: Neural Network Pruning with Combinatorial Optimization

This is the offical repo of the ICML 2023 paper **Fast as CHITA: Neural Network Pruning with Combinatorial Optimization**

## Requirements
This code has been tested with Python 3.7 and the following packages:
```
numba==0.56.4
numpy==1.21.6
scikit_learn==1.0.2
torch==1.12.1+cu113
torchvision==0.13.1+cu113
```

## Pruned models
We provide checkpoints for our best pruned models, obtained with the gradual pruning procedure described in the paper.

### MobileNetV1
|Sparsity|Checkpoint|
|--------|----------|
|75.28|[link](https://drive.google.com/file/d/1GcA2e9jO0Z-WdG5J-qEnJKGymQcuCeaV/view?usp=share_link)|
|89.00|[link](https://drive.google.com/file/d/1RilIWtf-1uM_iAY0R3nHajWAsztfSjOr/view?usp=share_link)|

### ResNet50
|Sparsity|Checkpoint|
|--------|----------|
|90.00|[link](https://drive.google.com/file/d/1Shhrd7Ck9lfFQzaRUVFrbj0dOPniTgYD/view?usp=share_link)|
|95.00|[link](https://drive.google.com/file/d/10Tth8fFVIssYKupQHTrM0P3tLKG2mT1t/view?usp=share_link)|
|98.00|[link](https://drive.google.com/file/d/1rsYM6OdtSnMPTRtYYojfSvTMdmAoPQBG/view?usp=share_link)|

## Structure of the repo
Scripts to run the algorithms are located in `scripts/`. The current code supports the following architectures (datasets): MLPNet (MNIST), ResNet20 (Cifar10), MobileNetV1 (Imagenet) and ResNet50 (Imagenet). Adding new models can be done through `model_factory` function in `utils/main_utils.py`. 


## Citing CHITA
If you find CHITA useful in your research, please consider citing the following paper.
```
@InProceedings{pmlr-v202-benbaki23a,
  title = 	 {Fast as {CHITA}: Neural Network Pruning with Combinatorial Optimization},
  author =       {Benbaki, Riade and Chen, Wenyu and Meng, Xiang and Hazimeh, Hussein and Ponomareva, Natalia and Zhao, Zhe and Mazumder, Rahul},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {2031--2049},
  year = 	 {2023},
}
```





