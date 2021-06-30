# Self domain adapted network
The source code for the paper: 

***Self domain adapted network*** ([MICCAI2020](https://arxiv.org/abs/2007.03162))    
***Autoencoder Based Self-Supervised Test-Time Adaptation for Medical Image Analysis***([MEDIA2021](https://www.sciencedirect.com/science/article/abs/pii/S1361841521001821))

## Enviroment setup (optional)
Download Nvidia docker for pytorch
```
docker pull nvcr.io/nvidia/pytorch:20.03-py3
```
Create a Dockerfile (already provided)
```
FROM nvcr.io/nvidian/pytorch:20.03-py3
RUN pip install nibabel
RUN pip install monai
RUN pip install scikit-image
RUN pip install tifffile
```
Then
```
docker build -f Dockerfile -t nvcr.io/nvidian/pytorch:20.03-py3-tt .
```
Run into Docker:
```
sudo docker run -it --gpus all --pid=host --shm-size 8G -v /home/yufan/Projects/:/workspace/Projects/ nvcr.io/nvidian/pytorch:20.03-py3-tt
```
## Dataset
OCTDataset ([download](https://drive.google.com/file/d/1gi6wR-m6s6UDhwLey1jP_uuwwtC8Py3x/view?usp=sharing))

MRIDataset (IXI [download](https://brain-development.org/ixi-dataset/), JHU currently unavailable due to privacy issue)
## Usage    
### Offline Training
1. Train Task model (segmentation/synthesis UNet) on source domain (on GPU 0). The datasets/dataset.py need to be modified for your own processed dataset. Also change the data and code path in the bash script.
```
bash ./scripts/train_oct.sh 0 tnet
```


2. Train Auto-encoders on source domain (on GPU 0). Change the tnet_checkpoint_XX.pth to the best validation model. Also change the data and code path in the bash script.
```
bash ./scripts/train_oct.sh 0 aenet
```
### Test-Time Adaptation
3. Test on target domain. The network is adapted to each single 3D test subject
with their names listed in a txt file --sub_name (subject_01, subject_02 xxx), and the 2D images are extracted and saved as png (subject_01_0.png, subject_01_1.png xxx)
in the --vimg_path folder. If convergence is not good, try add --sepochs=5 in the test_oct.sh
to stablize the convergence (pre-train the pixel-level adaptor).
```
bash ./scripts/test_oct.sh 0
```

## Citation
Please cite:
```
@inproceedings{he2020self,
  title={Self domain adapted network},
  author={He, Yufan and Carass, Aaron and Zuo, Lianrui and Dewey, Blake E and Prince, Jerry L},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={437--446},
  year={2020},
  organization={Springer}
}

@article{he2021autoencoder,
  title={Autoencoder Based Self-Supervised Test-Time Adaptation for Medical Image Analysis},
  author={He, Yufan and Carass, Aaron and Zuo, Lianrui and Dewey, Blake E and Prince, Jerry L},
  journal={Medical Image Analysis},
  pages={102136},
  year={2021},
  publisher={Elsevier}
}