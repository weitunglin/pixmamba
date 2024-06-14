<div align="center">
<h1>PixMamba </h1>
<h3>PixMamba: Leveraging State Space Models in a Dual-Level Architecture for Underwater Image Enhancement</h3>

</div>

- [Abstract](#abstract)
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Citation](#citation)
- [Acknowledgment](#acknowledgment)

## Abstract

Underwater Image Enhancement (UIE) is critical for marine research and exploration but hindered by complex color distortions and severe blurring. Recent deep learning-based methods have achieved remarkable results, yet these methods struggle with high computational costs and insufficient global modeling, resulting in locally under- or over- adjusted regions. We present PixMamba, a novel architecture, designed to overcome these challenges by leveraging State Space Models (SSMs) for efficient global dependency modeling. Unlike convolutional neural networks (CNNs) with limited receptive fields and transformer networks with high computational costs, PixMamba efficiently captures global contextual information while maintaining computational efficiency. Our dual-level strategy features the patch-level Efficient Mamba Net (EMNet) for reconstructing enhanced image feature and the pixel-level PixMamba Net (PixNet) to ensure fine-grained feature capturing and global consistency of enhanced image that were previously difficult to obtain. PixMamba achieves state-of-the-art performance across various underwater image datasets and delivers visually superior results.

## Overview

![arch](./arch.png)

## Getting Started

pixmamba implementation can be found [here](./classification/models/pixmamba.py).
training scripts integrated with [mmagic](https://github.com/open-mmlab/mmagic) will be released soon.

## Citation

```
@article{lin2024pixmamba,
    title={{PixMamba}: Leveraging State Space Models in a Dual-Level Architecture for Underwater Image Enhancement}, 
    author={Wei-Tung Lin and Yong-Xiang Lin and Jyun-Wei Chen and Kai-Lung Hua},
    journal={arXiv preprint arXiv:2406.08444},
    year={2024}
}
```

## Acknowledgment

This project is based on Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), [OpenMMLab](https://github.com/open-mmlab), [replknet](https://github.com/DingXiaoH/RepLKNet-pytorch/tree/main/erf), VMamba ([paper](https://arxiv.org/abs/2401.10166), [code](https://github.com/MzeroMiko/VMamba)), 
VM-UNet ([paper](https://arxiv.org/abs/2402.02491), [code](https://github.com/JCruan519/VM-UNet)), UIE_Benchmark ([code](https://github.com/ddz16/UIE_Benckmark)). Thanks for their excellent works.

## Contact

I'm happy to address any questions or concerns you may have. Please feel free to contact me at weitung8@gmail.com

