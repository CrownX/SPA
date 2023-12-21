# Official implementation for **SPA** (NeurIPS 2023)
This repository provides the code implementation for our paper [**[NeurIPS 2023] SPA: A Graph Spectral Alignment Perspective for Domain Adaptation**](https://arxiv.org/pdf/2310.17594.pdf). 

### Introduction
Unsupervised domain adaptation (UDA) is a pivotal form in machine learning to extend the in-domain model to the distinctive target domains where the data distributions differ. Most prior works focus on capturing the inter-domain transferability but largely overlook rich intra-domain structures, which empirically results in even worse discriminability. In this work, we introduce a novel graph SPectral Alignment (**SPA**) framework to tackle the tradeoff. The core of our method is briefly condensed as follows: (i)-by casting the DA problem to graph primitives, SPA
composes a coarse graph alignment mechanism with a novel spectral regularizer towards aligning the domain graphs in eigenspaces; (ii)-we further develop a fine-grained message propagation module — upon a novel neighbor-aware self-training mechanism — in order for enhanced discriminability in the target domain. On standardized benchmarks, the extensive experiments of SPA demonstrate that its performance has surpassed the existing cutting-edge DA methods. Coupled with dense model analysis, we conclude that our approach indeed possesses superior efficacy, robustness, discriminability, and transferability. 

### Citation
If this code is useful for your research, please refer to this paper for more details:

```
@inproceedings{xiao2023spa,
  title={SPA: A Graph Spectral Alignment Perspective for Domain Adaptation},  
  author={Xiao, Zhiqing and Wang, Haobo and Jin, Ying and Feng, Lei and Chen, Gang and Huang, Fei and Zhao, Junbo},
  journal={NeurIPS},
  year={2023}
}
```


### Training

1. install packages

   `python == 3.7.13`
   `pytorch == 1.10.1`
   `torchvision == 0.11.2`
   `sklearn == 1.0.2`
   `numpy == 1.12.6`
   `argparse, random, PIL`
   
2. download dataset

    * Office31
    * OfficeHome
    * VisDA2017
    * DomainNet


3. run the train file with 'train_[uda/ssda]_[dataset].sh', e.g.

   `sh train_uda_officehome.sh spa 0.2 1.0 DANNE 0.3 laplac1 gauss `
