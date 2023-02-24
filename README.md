# Source-Free Active Domain Adaptation via Energy-Based Locality Preserving Transfer
- Pytorch implementation for **Source-Free Active Domain Adaptation via Energy-Based Locality Preserving Transfer**, https://doi.org/10.1145/3503161.3548152.
- This work firstly combines Active Domain Adaptation (ADA) and Source Free Domain Adaptation (SFDA), proposing a new setting Source Free Active Domain Adaptation (SFADA). Furthermore, we propose a Locality Preserving Transfer (LPT) framework to achieve adaptation without source data. Meanwhile, a label propagation strategy is adopted to improve the discriminability for better adaptation. After LPT, unique samples are identified by an energy-based approach for active annotation. Finally, with supervision from the annotated samples and pseudo labels, a well adapted model is obtained. Extensive experiments on three widely used UDA benchmarks show that our method is comparable or superior to current state-of-the-art active domain adaptation methods even without access to source data.
![1677164300392](https://user-images.githubusercontent.com/68037940/220943650-094eec69-b633-4696-908d-5848a1b858eb.png)

# Environment
`pytorch==1.10`

# Dataset preparation
- Download VisDA dataset into `data/train/` (source domain) and `data/validation/` (target domain).

# Usage
1. **Train source model**: `python train_src.py`
2. **Perform source-free active domain adaptation**: `python train_tar.py`

# Thanks
- We use code from *Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation* and *Do we really need to access the
source data? source hypothesis transfer for unsupervised domain adaptation* in our code. Thanks for their great work!

# Citation
```
@inproceedings{li2022source,
  title={Source-Free Active Domain Adaptation via Energy-Based Locality Preserving Transfer},
  author={Li, Xinyao and Du, Zhekai and Li, Jingjing and Zhu, Lei and Lu, Ke},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={5802--5810},
  year={2022}
}
```
