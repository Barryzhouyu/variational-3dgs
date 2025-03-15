# Variational Multi-scale Representation for Estimating Uncertainty in 3D Gaussian Splatting
[Ruiqi Li](https://www.comp.hkbu.edu.hk/~csrqli/), [Yiu-ming Cheung](https://www.comp.hkbu.edu.hk/~ymc/) <br>
[Paper Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/a076d0d1ed77364fc57693bdee1958fb-Paper-Conference.pdf) 

This repository contains the official open-source implementation of the paper "Variational Multi-scale Representation for Estimating Uncertainty in 3D Gaussian Splatting". We developed an uncertainty estimation method for Gaussian Splatting method based on Bayesian inference framework and multi-scale representation. 

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

The LF dataset and LLFF dataset files are provided here: [LF dataset](https://drive.google.com/file/d/1RrfrMN5wSaishYJu5vYiTy6gUPZfLaDM/view?usp=sharing), [LLFF dataset](https://drive.google.com/file/d/1kDclWpEpUPm9Nw0tGoQTLWz3L4g5Hu2L/view?usp=sharing). 

Please unzip and put them under the a dataset folder: 

```bash
variational-gs
│
├──dataset
│   │  
│   ├──── LF
│   └──── nerf_llff_data
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

