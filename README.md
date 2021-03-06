# Adversarial Separation Network for Cross-Network Node Classification
This repository contains the author's implementation in Pytorch for the paper "Adversarial Separation Network for Cross-Network Node Classification".
<div align = "center">
<img src="https://z3.ax1x.com/2021/08/16/fWMeCq.png" width = "85%"/>
<br>
<!-- <div>figure1 The framework ofAdversarial Separation Network(ASN) for cross-network node classification</div> -->
<br>
</div>


## Environment Requirement

* Python: 3.6

* PyTorch: 1.5.1 (with suitable CUDA and CuDNN version)

* tensorboard: 2.3.0

* Scipy:1.2.1

* Numpy:1.16.2

* sklearn:0.21.1


## Datasets:

The data folder includes different domain data. 

The preprocessed data can be found in our repository.

* `/data/acmv9.mat`

* `/data/dblpv7.mat`

* `/data/citationv1.mat`

The orginal datasets can be founded from https://www.aminer.cn/citation.


##  Training:

You can run the command in run.sh to train and evaluate on each task for network graph dataset. 

Before that, you need to change the data_root (data root path), learning rate and cuda (gpu options) in the script.

> python main.py --data_src 'dblpv7' --data_trg 'acmv9' --lr 3e-2 --cuda '0'

## Results
<div align = "center">
<img src="https://z3.ax1x.com/2021/08/16/fWQ1Qf.png" width = "80%"/>
<br>
<!-- <div>Table 3: Node classification accuracy comparisons on six cross-domain tasks</div> -->
<br>
</div>


## Contact 
- zhangxw@smail.nju.edu.cn
- duyuntao@smail.nju.edu.cn

## Reference
 
