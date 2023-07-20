# DeepCPD
Article: Neural-Network-Based Pose Estimation During Noncooperative Spacecraft Rendezvous Using Point Cloud

## Prerequisites:
PyTorch 1.8.2  
open3d  
h5py  
numpy  
tqdm  
TensorboardX  

## Training
python main.py

## Testing
python main.py --eval  
Refined by ICP with batch size = 1：    
python main.py --eval --icp  

## Citation
Please cite this paper if you want to use it in your work,  
@article{doi:10.2514/1.I011179,
author = {Zhang, Shaodong and Hu, Weiduo and Guo, Wulong and Liu, Chang},
title = {Neural-Network-Based Pose Estimation During Noncooperative Spacecraft Rendezvous Using Point Cloud},
journal = {Journal of Aerospace Information Systems},
volume = {0},
number = {0},
pages = {1-11},
year = {2023},
doi = {10.2514/1.I011179}}