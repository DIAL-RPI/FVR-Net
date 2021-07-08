# Frame-to-Volume Registration Network (FVR-Net)
This repository contains the source code for MICCAI-2021 paper, entitled "End-to-end Ultrasound Frame to Volume Registration". 
![FVR-Net Architecture](figures/FVR-Net.jpg)

## Introduction
In this work, the proposed FVR-Net can be trained to automatically register a single transrectal ultrasound (TRUS) 2D frame to a reconstructed 3D TRUS volume, which potentially enables instant frame localization during the prostate biopsy. The proposed FVR-Net utilizes a dual-branch feature extraction module to extract the information from TRUS frame and volume to estimate transformation parameters. To achieve efficient training and inference, we introduce a differentiable 2D slice sampling module which allows gradients backpropagating from an unsupervised image similarity loss for content correspondence learning. We include the training codes for the current version, and the repository is under active update.

## Environment
- Set up your environment by anaconda, (**python3.7, torch 1.5.0+cu92**)
