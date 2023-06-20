# Holistic-Super-Resolution-Review
Review of deep-learning based super-resolution method in different fields.  

:blush: The wealth of the literature on SR is so rich that we could not give an exhaustic review. We just list the major methods along the timeline.  

## :heavy_exclamation_mark: Single Image Super-Resolution (SISR)  
We categorize the SISR into three types, [**Convolutional Neural Network-based**](./Single-Image-Super-Resolution/Convolutional-Neural-Network-based.md), [**Generative Adversarial Network-based**](./Single-Image-Super-Resolution/Generative-Adversarial-Network-based.md) and [**Transformer-based**](./Single-Image-Super-Resolution/Transformer-based.md), for their purpose.  

:boom: **Datasets**  
:one: [Set5](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u)   
:two: [Set14](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u)     
:three: [BSD100](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u)     
:four: [Urban100](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u)     
:five: [Manga109](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u)     
:six: [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar)  
:seven: [Flickr1024](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)  

:dart: **Experimental Results on X4 Task**
Model | Params (K) | Set5 | Set14 | BSD100 | Urban100 | Training Set/s |
:-:|:-:|:-:|:-:|:-:|:-:|:-:
Bicubic   | - | 28.43/0.8109 | 26.00/0.7023 | 25.96/0.6678 | 23.14/0.6574 | - 
TPSR-D2 | 61 | 29.60/- | 26.88/- | 26.23/- | 24.12/- | DIV2K 
LUT | 1274 | 29.82/0.8478| 27.01/0.7355 | 26.53/0.6953 | 24.02/0.6990 | DIV2K 
SRCNN | 24 | 30.48/0.8628 | 27.50/0.7513 | 26.90/0.7103 | 24.52/0.7226 | T91+ImageNet 
ESPCN | 25 | 30.52/0.8697 | 27.42/0.7606 | 26.87/0.7216 | 24.39/0.7241 | T91+ImageNet 
SPLUT | 18000 | 30.52/0.8630 | 27.54/0.7520 | 26.87/0.7090 | 24.46/0.7190 | DIV2K
RankSRGAN | - | - | 26.57/- | 25.57/- | - | DIV2K+Flickr2K  
FSRCNN | 12 | 30.70/0.8657 | 27.59/0.7535 | 26.96/0.7128 | 24.60/0.7258 | T91+General-100 
CSCN | - | 30.86/0.8732 | 27.64/0.7578 | 27.03/0.7161 | - | T91
NatSR | 4800 | 30.98/0.8606 | 27.42/0.7329 | 26.44/0.6827 | 25.46/0.7602 | DIV2K
ZSSR | 225 | 31.13/0.8796 | 28.01/0.7651 | 27.12/0.7211 | - | -
VDSR | 665 | 31.35/0.8838 | 28.01/0.7674 | 27.29/0.7251 | 25.18/0.7524 | BSD+T91
DSRN | 1200 | 31.40/0.8830 | 28.07/0.7700 | 27.25/0.7240 | 25.08/0.7470 | T91
DRCN | 1775 | 31.53/0.8854 | 28.02/0.7670 | 27.23/0.7233 | 25.14/0.7510 | T91
SESR | 115 | 31.54/0.8866 | 28.12/0.7712 | 27.31/0.7277 | 25.31/0.7604 | DIV2K 
LapSRN | 812 | 31.54/0.8850 | 28.19/0.7720 | 27.32/0.7270 | 25.21/0.7560 | BSD+T91 
DRRN | 207 | 31.68/0.8888 | 28.21/0.7720 | 27.38/0.7284 | 25.44/0.7638 | BSD+T91
ENet-PAT | - | 31.74/- | 28.42/- | 27.50/- | 25.66/- | MSCOCO
MemNet | 677 | 31.74/0.8893 | 28.26/0.7723 | 27.40/0.7281 | 25.50/0.7630 | BSD+T91 
IDN | 678 | 31.82/0.8903 | 28.25/0.7730 | 27.41/0.7297 | 25.41/0.7632 | BSD+T91
SRResNet | 1500 | 31.92/0.8998 | 28.39/0.8166 | 27.52/0.7603 | - | ImageNet 
NAPS | 125 | 31.93/0.8906 | 28.42/0.7763 | 27.44/0.7307 | 25.66/0.7715 | DIV2K
SRMDNF | - | 31.96/0.8930 | 28.35/0.7770 | 27.49/0.7340 | 25.68/0.7730 | BSD+DIV2K+WED
SRDenseNet | 5452 | 32.02/0.7819 | 28.50/0.7782 | 27.53/0.7337 | 26.05/0.7819 | ImageNet 
MSRN | 6300 | 32.07/0.8903 | 28.60/0.7751 | 27.52/0.7273 | 26.04/0.7896 | DIV2K
SMSR | 1006 | 32.12/0.8932 | 28.55/0.7808 | 27.55/0.7351 | 26.11/0.7868 | DIV2K 
DCLS | - | 32.12/0.8890 | 28.54/0.7728 | 27.60/0.7285 | 26.15/0.7809 | DIV2K+Flickr2K
EDSR-SLS | 363 | - | 28.49/- | 27.51/- | 25.84/- | DIV2K
CARN | 1600 | 32.13/0.8937 | 28.60/0.7806 | 27.58/0.7349 | 26.07/0.7837 | BSD+T91+DIV2K 
ESRT | 751 | 32.19/0.8947 | 28.69/0.7833 | 27.69/0.7379 | 26.39/0.7962 | DIV2K 
IMDN | 715 | 32.21/0.8948 | 28.58/0.7811 | 27.56/0.7353 | 26.04/0.7838 | DIV2K 
SRFeat | 6189 | 32.27/0.8938 | 28.71/0.7835 | 27.64/0.7378 | - | DIV2K
LatticeNet | 777 | 32.30/0.8962 | 28.68/0.7830 | 27.62/0.7367 | 26.25/0.7873 | DIV2K
RDN-MetaSR | - | 32.38/- | 28.78/- | 27.71/- | 26.55/- | DIV2K
SCN | 1200 | 32.39/0.8981 | 28.74/0.7869 | 27.69/0.7415 | 26.50/0.8000 | DIV2K 
SRFBN | 3500 | 32.46/0.8968 | 28.80/0.7876 | 27.71/0.7420 | 26.64/0.8033 | DIV2K+Flickr2K 
EDSR | 43000 | 32.46/0.8968 | 28.80/0.7876 | 27.71/0.7420 | 26.64/0.8033 | DIV2K 
RDN | 21900 | 32.47/0.8990 | 28.81/0.7871 | 27.72/0.7419 | 26.61/0.8028 | DIV2K 
DBPN | 10000 | 32.47/0.8980 | 28.82/0.7860 | 27.72/0.7400 | 26.38/0.7946 | DIV2K+Flickr2K 
RNAN | 225 | 32.49/0.8982 | 28.83/0.7878 | 27.72/0.7421 | 26.61/0.8023 | DIV2K 
RDN-LIIF | - | 32.50/- | 28.80/- | 27.74/- | 26.68/- | DIV2K
OISR | 15592 | 32.53/0.8992 | 28.86/0.7878 | 27.75/0.7428 | 26.79/0.8068 | DIV2K 
ArbRCAN | 16600 | 32.55/- | 28.87/- | 27.76/- | 26.68/- | DIV2K
NLSN | - | 32.59/0.9000 | 28.87/0.7891 | 27.78/0.7444 | 26.96/0.8109 | DIV2K
RCAN | 16000 | 32.63/0.9002 | 28.87/0.7889 | 27.77/0.7436 | 26.82/0.8087 | DIV2K 
SAN | 15700 | 32.64/0.9003 | 28.92/0.7888 | 27.78/0.7436 | 26.79/0.8068 | DIV2K 
HAN | 64199 | 32.64/0.9002 | 28.90/0.7890 | 27.80/0.7442 | 26.85/0.8094 | DIV2K 
IPT | 115000 | 32.64/- | 29.01/- | 27.82/- | 27.26/- | ImageNet 
FAD-RCAN | - | 32.65/0.9007 | 28.88/0.7889 | 27.78/0.7437 | 26.86/0.8092 | DIV2K
RFANet | 11000 | 32.66/0.9004 | 28.88/0.7894 | 27.79/0.7442 | 26.92/0.8112 | DIV2K 
CSNLN | 30000 | 32.68/0.9004 | 28.95/0.7888 | 27.80/0.7439 | 27.12/0.8168 | DIV2K
CRAN | 19940 | 32.72/0.9012 | 29.01/0.7918 | 27.86/0.7460 | 27.13/0.8167 | DIV2K
DRN | 9800 | 32.74/0.9020 | 28.98/0.7920 | 27.83/0.7450 | 27.03/0.8130 | DIV2K+Flickr2K 
ELAN | 8312 | 32.75/0.9022 | 28.96/0.7914 | 27.83/0.7459 | 27.13/0.8167 | DIV2K 
EBRN | 7900 | 32.79/0.9032 | 29.01/0.7903 | 27.85/0.7464 | 27.03/0.8114 | DIV2K
DFSA | - | 32.79/0.9019 | 29.06/0.7922 | 27.87/0.7458 | 27.17/0.8163 | DIV2K+Flickr2K 
SwinIR-LTE | - | 32.81/- | 29.06/- | 27.86/- | 27.24/- | DIV2K
SwinIR | 11800 | 32.92/0.9044 | 29.09/0.7950 | 27.92/0.7489 | 27.45/0.8254 | DIV2K+Flickr2K 

## :heavy_exclamation_mark: Video Super-Resolution (VSR)  
We categorize the SISR into three types, [**Convolutional Neural Network-based**](./Video-Super-Resolution/Convolutional-Neural-Network-based.md), and [**Transformer-based**](./Video-Super-Resolution/Transformer-based.md), for their purpose.  

:boom: **Datasets**  
:one: [REDS](https://seungjunnah.github.io/Datasets/reds.html)   
:two: [Video-90K](http://toflow.csail.mit.edu/)     
:three: [Vid4](https://drive.google.com/file/d/1ZuvNNLgR85TV_whJoHM7uVb-XW1y70DW/view)     
:four: [UDM10](https://www.terabox.com/web/share/link?surl=LMuQCVntRegfZSxn7s3hXw&path=%2Fproject%2Fpfnl)     
:five: [UDF](https://github.com/yhjo09/VSR-DUF/tree/master/inputs)     
:six: [SPMC](https://tinyurl.com/y426dcn9)  

:dart: **Experimental Results on X4 Task**
Model | Params (M) | Vid4 | Video-90K | REDS | Training Set/s |
:-:|:-:|:-:|:-:|:-:|:-:
Bicubic | - | 23.78/0.6347 | 31.32/0.8684 | 23.72/0.7559 | - 
VESPCN | - | 25.35/0.7557 | -/- | 24.93/0.8107 | CDVL
FRVSR | 5.1 | 26.69/0.8220 | -/- | 25.27/0.8256 | Vimeo-90K 
SPMC | - | 25.88/0.7752 | -/- | -/- | SPMCS 
DUF | 5.8 | 27.33/0.8319 | 36.37/0.9387 | 28.63/0.8251 | DUF 
RBPN | 12.2 | 27.12/0.8180 | 37.07/0.9435 | 25.17/0.8187 | Vimeo-90K 
PFNL | 3.0 | 26.73/0.8029 | 36.14/0.9363 | 29.63/0.8502 | UDM10 
VSR\_TGA | - | 27.59/0.8419 | -/- | -/- | Vimeo-90K 
MuCAN | - | -/- | 37.32/0.9465 | 30.88/0.8750 | REDS+Vimeo-90K 
RSDN | 6.2 | 27.79/0.8474 | 37.05/0.9454 | -/- | Vimeo-90K 
BasicVSR | 6.3 | 27.24/0.8251 | 37.18/0.9450 | 31.42/0.8909 | REDS+Vimeo-90K 
IconVSR | 8.7 | 27.39/0.88279 | 37.47/0.9476 | 31.67/0.8948 | REDS+Vimeo-90K 
DSMC | - | -/- | -/- | 25.73/0.8428 | REDS 
VSRT | 32.6 | 27.36/0.8258 | 37.71/0.9494 | 25.73/0.8428 | REDS+Vimeo-90K 
VRT | 35.6 | 27.93/0.8425 | 38.20/0.9530 | 25.73/0.8428 | REDS 

## :heavy_exclamation_mark: Stereo Super-Resolution (SSR)  
We categorize the SSR into three types, [**Convolutional Neural Network-based**](./Stereo-Super-Resolution/Convolutional-Neural-Network-based.md), [**Generative Adversarial Network-based**](./Stereo-Super-Resolution/Generative-Adversarial-Network-based.md) and [**Transformer-based**](./Stereo-Super-Resolution/Transformer-based.md), for their purpose.  

:boom: **Datasets**  
:one: [Middlebury](https://vision.middlebury.edu/stereo/data/)   
:two: [Tsukuba](https://home.cvlab.cs.tsukuba.ac.jp/dataset)     
:three: [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)     
:four: [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)     
:five: [Flickr1024](https://yingqianwang.github.io/Flickr1024/)     

:dart: **Experimental Results on X4 Task**
Model | Params (M) | KITTI2015 | KITTI2012 | Middlebury | Training Set/s |
:-:|:-:|:-:|:-:|:-:|:-:
Bicubic  | -  | 23.90/0.7100  | 24.64/0.7334  | 26.39/0.7564  |  - 
StereoSR  | 1.06  | 25.12/0.7679  | 25.94/0.7839  | 28.24/0.8133  | Middlebury+KITTI+Tsukuba 
PASSRnet  | 1.35  | 25.34/0.7722  | 26.18/0.7874  | 28.36/0.8153  | Middlebury+Flickr1024 
DASSR  | 1.1  | 25.35/0.8740  | 26.96/0.8820  | 29.83/0.9090  | Flickr1024 
SRRes+SAM  | 1.73  | 25.55/0.7825  | 26.35/0.7957  | 28.76/0.8287  | Flickr1024 
CPASSRNet  | 42.39  | 25.12/0.7693  | 25.31/0.7712  | 28.31/0.8194  | Middlebury 
iPASSR  | 1.42  | 25.61/0.7850  | 26.47/0.7993  | 29.07/0.8363  | Middlebury+Flickr1024 
SSRDE-FNet  | 2.24  | 25.74/0.7884  | 26.61/0.8028  | 29.29/0.8407  | Flickr1024 
IMSSRnet  | 6.89  | 25.59/-  | 26.44/-  | 29.02/-  | Middlebury+Flickr1024 
CVCNet  | 0.99  | 25.55/0.7801  | 26.35/0.7935  | 28.65/0.8231  | Flickr1024 

## :heavy_exclamation_mark: Light Field Super-Resolution (LFSR)  
We categorize the LFSR into three types, [**Convolutional Neural Network-based**](./Light-Field-Super-Resolution/Convolutional-Neural-Network-based.md), [**Generative Adversarial Network-based**](./Light-Field-Super-Resolution/Generative-Adversarial-Network-based.md) and [**Transformer-based**](./Light-Field-Super-Resolution/Transformer-based.md), for their purpose.  

:boom: **Datasets**  
:one: [EPFL](https://infoscience.epfl.ch/record/218363)  
:two: [HCInew](https://lightfield-analysis.uni-konstanz.de/)     
:three: [HCIold](https://lightfield-analysis.uni-konstanz.de/)     
:four: [INRIA](https://pan.baidu.com/s/19iGLK57mXqC4_g8idEQovg) | Code：pkyv   
:five: [STFgantry](https://pan.baidu.com/s/1WRTh_AYu_H9kR-DqSBb2dg) | Code：qjwv     

:dart: **Experimental Results on X4 Task**
Model | Params (M) | EPFL | HCInew | HCIold | INRIA | STFgantry |
:-:|:-:|:-:|:-:|:-:|:-:|:-:
Bicubic | -  | 25.14/0.8311  | 27.61/0.8507  | 32.42/0.9335  | 26.82/0.8860  | 25.93/0.8431 
VDSR  | 0.67  | 27.25/0.8782  | 29.31/0.8828  | 34.81/0.9518  | 29.19/0.9208  | 28.51/0.9012 
EDSR | 12  | 27.84/0.8858  | 29.60/0.8874  | 35.18/0.9538  | 29.66/0.9259  | 28.70/0.9075 
RCAN  | 25  | 27.88/0.8863  | 29.63/0.8880  | 35.20/0.9540  | 29.76/0.9273  | 28.90/0.9110 
resLF  | 6.79  | 27.46/0.8899  | 29.92/0.9011  | 36.12/0.9651  | 29.64/0.9339  | 28.99/0.9214 
LFSSR  | 1.16  | 28.27/0.9080  | 30.72/0.9124  | 36.70/0.9690  | 30.31/0.9446  | 30.15/0.9385 
LF-ATO  | 1.36  | 28.25/0.9120  | 30.88/0.9140  | 37.00/0.9700  | 30.71/0.9490  | 30.61/0.9430 
MEG-Net  | 1.77  | 28.74/0.9160  | 31.10/0.9180  | 37.28/0.9720  | 30.66/0.9490  | 30.77/0.9450 
LF-InterNet  | 5.23  | 28.67/0.9143  | 30.98/0.9165  | 37.11/0.9715  | 30.64/0.9486  | 30.53/0.9426 
LF-IINet  | 4.89  | 29.11/0.9200  | 31.36/0.9210  | 37.62/0.9740  | 31.08/0.9520  | 31.21/0.9500 
LF-DFnet  | 3.99  | 28.77/0.9165  | 31.23/0.9196  | 37.32/0.9718  | 30.83/0.9503  | 31.15/0.9494 
DPT  | 3.78  | 28.93/0.9167  | 31.19/0.9186  | 37.39/0.9720  | 30.96/0.9502  | 31.14/0.9487 
LFT  | 1.16  | 29.25/0.9210  | 31.46/0.9220  | 37.63/0.9740  | 31.20/0.9520  | 31.86/0.9550 
DistgSSR | 3.58  | 28.98/0.9190  | 31.38/0.9220  | 37.55/0.9730  | 30.99/0.9520  | 31.63/0.9530 
