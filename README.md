# Self supervised adaptation
The source code for the paper: 

***Self domain adapted network*** ([MICCAI2020](https://arxiv.org/abs/2007.03162))
## Getting Started
1. Train Task model (segmentation/synthesis UNet) on source domain
```
python train.py 
--epochs=20 # total training epochs
--task=seg # segmentation unet. choices[seg, syn]
--batch-size=2 
--img_path=../data/octdata/hctrain/image/ # training image folder
--label_path=../data/octdata/hctrain/label/ # training label folder
--vimg_path=../data/octdata/hceval/image/ # validation image folder
--vlabel_path=../data/octdata/hceval/label/ # validation label folder
--trainer=tnet # train task model. choices[tnet, aenet]
--img_ext=png # image extension
--label_ext=txt # label extension
--results_dir=../exps/seg_tnet/ # folders to save results
--ss=1 # save step
```
The datasets/dataset.py need to be modified for your own processed dataset.

2. Train Auto-encoders on source domain
```
python train.py ^
--epochs=20 
--task=seg 
--batch-size=2
--img_path=../data/octdata/hctrain/image/
--label_path=../data/octdata/hctrain/label/ 
--vimg_path=../data/octdata/hceval/image/ 
--vlabel_path=../data/octdata/hceval/label/
--trainer=aenet 
--img_ext=png 
--label_ext=txt
--results_dir=../exps/seg_tnet/
--wt=1,0,1,1,1,1 # use 5 autoencoders
--ss=1 
--resume_T=../exps/seg_tnet/checkpoints/your_best_t_model.pth
```

3. Test on target domain. The network is adapted to each single 3D test subject
with their names listed in a txt file --sub_name (subject_01, subject_02 xxx), and the 2D images are extracted and saved as png (subject_01_0.png, subject_01_1.png xxx)
in the --vimg_path folder. If convergence is not good, try add --sepochs=5
to stablize the convergence.
```
python train.py 
--tepochs=5 # adaptation epochs
--task=seg
--batch-size=1
--vimg_path=../data/octdata/cirrus/image/ # test image path
--vlabel_path=../data/octdata/cirrus/label/ # test label path (validation only)
--sub_name=../data/octdata/cirrus/cirrus_name.txt # subject name list
--trainer=aenet
--img_ext=png
--label_ext=txt
--results_dir=../exps/seg_tnet/
--wt=1,0,1,1,1,1 # use 5 autoencoders
--wo=1 # orthogonality loss weight
--seq=1,2,3 # use feature adaptors 1,2,3
--dpi=300 # image dpi
--si # save image
--resume_T=../exps/seg_tnet/checkpoints/your_best_t_model.pth
--resume_AE=../exps/seg_tnet/checkpoints/your_best_ae_model.pth
-t # testing stage
```

### Citation
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