# Variational Multi-scale Representation for Estimating Uncertainty in 3D Gaussian Splatting
[Ruiqi Li](https://www.comp.hkbu.edu.hk/~csrqli/), [Yiu-ming Cheung](https://www.comp.hkbu.edu.hk/~ymc/) <br>
[Paper Link](https://openreview.net/pdf?id=qpeAtfUWOQ) 

This repository contains the official open-source implementation of the paper "Variational Multi-scale Representation for Estimating Uncertainty in 3D Gaussian Splatting". 

Abstract: *Recently, 3D Gaussian Splatting (3DGS) has become popular in reconstructing dense 3D representations of appearance and geometry. However, the learning pipeline in 3DGS inherently lacks the ability to quantify uncertainty, which is an important factor in applications like robotics mapping and navigation. In this paper, we propose an uncertainty estimation method built upon the Bayesian inference framework. Specifically, we propose a method to build variational multi-scale 3D Gaussians, where we leverage explicit scale information in 3DGS parameters to construct diversified parameter space samples. We develop an offset table technique to draw local multi-scale samples efficiently by offsetting selected attributes and sharing other base attributes. Then, the offset table is learned by variational inference with multi-scale prior. The learned offset posterior can quantify the uncertainty of each individual Gaussian component, and be used in the forward pass to infer the predictive uncertainty. Extensive experimental results on various benchmark datasets show that the proposed method provides well-aligned calibration performance on estimated uncertainty and better rendering quality compared with the previous methods that enable uncertainty quantification with view synthesis. Besides, by leveraging the model parameter uncertainty estimated by our method, we can remove noisy Gaussians automatically, thereby obtaining a high-fidelity part of the reconstructed scene, which is of great help in improving the visual quality. *


## Citation
If you found our work useful welcome to cite our paper: 

```
@inproceedings{li2024variational,
  title={Variational Multi-scale Representation for Estimating Uncertainty in 3D Gaussian Splatting},
  author={Li, Ruiqi and Cheung, Yiu-ming},
  booktitle={Advances in Neural Information Processing Systems},
  volume={37},
  pages={87934--87958},
  year={2024}
}
```

## Requirements

**Hardware Requirements**

CUDA-ready GPU with Compute Capability 7.0+

**Software Requirements**

Conda (recommended for easy setup)

C++ Compiler for PyTorch extensions

CUDA SDK 11 for PyTorch extensions

C++ Compiler and CUDA SDK must be compatible

## Usage

### Cloning the Repository

Please clone with submodules: 
```shell
# SSH
git clone git@github.com:csrqli/variational-3dgs.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/csrqli/variational-3dgs --recursive
```

### Setup

We provide conda environment file to creat experiment environment: 
```shell
conda env create --file environment.yml
conda activate variational_gs
```
We test our code on ubuntu system, please refer to original 3DGS repo about the potential error building the environment or running on windows. 

### Preparing Dataset

The LF dataset and LLFF dataset files is provided here: [LF dataset](https://drive.google.com/file/d/1RrfrMN5wSaishYJu5vYiTy6gUPZfLaDM/view?usp=sharing), [LLFF dataset](https://drive.google.com/file/d/1kDclWpEpUPm9Nw0tGoQTLWz3L4g5Hu2L/view?usp=sharing). 

Please unzip and put them under the a dataset folder: 

```bash
variational-gs
│
├──dataset
│   │  
│   ├──── LF
│   └──── LLFF
```

### Running

To train and evaluate the image quality and the image/depth uncertainty on LF dataset: 

```shell
python train.py --eval --dataset_name LF -s ./dataset/LF/$scene_name --resolution 2 --iterations 3000 --densify_until_iter 2000 --model_path ./output/$scene_name
```

To get the averaged results: 
```
python stat.py --dataset_name LF
```

To train and evaluate the image quality and image uncertainty quality on LLFF dataset: 

```shell
python train.py --eval --dataset_name LLFF -s ./dataset/nerf_llff_data/$scene_name --resolution 8 --iterations 7000 --densify_until_iter 4000 --model_path ./output/$scene_name
```

and also get the averaged results: 
```
python stat.py --dataset_name LLFF
```

## Funding and Acknowledgments

This work was supported in part by the NSFC / Research Grants Council (RGC) Joint Research Scheme under the grant: N\_HKBU214/21, and the RGC Senior Research Fellow Scheme under the grant: SRFS2324-2S02. 

