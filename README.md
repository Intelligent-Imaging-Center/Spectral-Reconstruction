# Learnable Reconstruction Methods from RGB Images to Hyperspectral Imaging: A Survey

A list of papers and resources for spectral reconstruction from images.

## Contents
1. [Introduction](#Introduction)
2. [Overview](#Overview)
3. [Datasets](#Datasets)
4. [Algorithm-Survey](#Algorithm-Survey)
5. [Results](#Results)
6. [References](#References)


## Introduction
Hyperspectral imaging enables versatile applications for its competence in capturing abundant spatial and spectral information, which is crucial for identifying substances. 
However, the devices for acquiring hyperspectral images are expensive and complicated.
Therefore, many alternative spectral imaging methodshave been proposed by directly reconstructing the hyperspectral  information  from  cost-effective  RGB  im-ages. 
We present a thorough investigation of more than25 state-of-the-art spectral reconstruction methods with respect to available database and corresponding evaluation criteria (e.g., reconstruction accuracy and quality).
It has revealed that, prior-based methods are suitable for small datasets, while deep learning can unleash its potential under large datasets. 
In future, with the expansion  of  datasets  and  the  design  of  more  advanced neural networks, deep learning methods with higher fea-ture representation abilities are more  promising.
This comprehensive review can serve as a fruitful reference source for peer researchers, thus paving the way for the development of most promising learnable methods.

![fig1](https://github.com/surunmu/Spectral_Reconstruction/blob/main/Figs/fig1.png)
Schematics of a common RGB camera and a hyperspectral imager 


## Overview
![fig2](https://github.com/surunmu/Spectral_Reconstruction/blob/main/Figs/fig2.png)
The overall taxonomy of the spectral reconstruction methods and the full lists for each category


## Datasets

| Dataset        | Amount           | Resolution  |Spectral range/(nm)  | Scene  |
|:----------------:|:-----------------:|:------------------:|:----------------:|:-------------:|
| [CAVE](https://www.cs.columbia.edu/CAVE/databases/multispectral/)| 32 | <img src="https://latex.codecogs.com/png.image?\dpi{110}&space;512\times&space;512\times&space;31" title="512\times 512\times 31" />|400-700|studio images of various objects|
| [ICVL](http://icvl.cs.bgu.ac.il/hyperspectral/)| 203|  <img src="https://latex.codecogs.com/png.image?\dpi{110}&space;1392\times&space;1300\times&space;31" title="1392\times 1300\times 31" />  |400-700|urban, suburban, rural, indoor and plant-life|
| [BGU-HS](https://competitions.codalab.org/competitions/18034#participate-get-data)| 286  |   <img src="https://latex.codecogs.com/png.image?\dpi{110}&space;1392\times&space;1300\times&space;31" title="1392\times 1300\times 31" /> |400-700|urban, suburban, rural, indoor and plant-life|
|[ARAD-HS](https://competitions.codalab.org/competitions/22225#participate)| 510  |    <img src="https://latex.codecogs.com/png.image?\dpi{110}&space;512\times&space;482\times&space;31" title="512\times 482\times 31" />  |400-700|various scenes and subjects|

## Algorithm-Survey 

### Prior-based methods
|Method   |Category     |Priors   |
|:-------------:|:--------------:|:--------------------:|
|Sparse Coding|Dictionary Learning|sparsity|
|SR A+|Dictionary Learning|sparsity, local euclidean linearity|
|Multiple Non-negative Sparse Dictionaries|Dictionary Learning|spatial structure similarity, spectral correlation|
|Local Linear Embedding Sparse Dictionary|Dictionary Learning|color and texture, local linearity|
|Spatially Constrained Dictionary Learning |Dictionary Learning|color and texture, local linearity|
|SR Manifold Mapping|Manifold Learning|ow-dimensional manifold|
|SR Gaussian Process |Gaussian Process|spectral physics, spatial structure similarity|

### Data-driven methods
#### Linear CNN
<div align="center">
<img src=Figs/fig3.png>
</div>
Three typical Linear CNN methods. (a) HSCNN. (b) SR2D/3DNet. (c) Residual HSRCNN


##### U-Net model
<div align="center">
<img src=Figs/fig4.png>
</div>
Spectral reconstruction methods using the U-Net model. (a) SRUNet. (b) SRMSCNN. (c) SR-MXRUNet. (d) SRBFWU-Net with supervised learning (left) and unsupervised learning (right).  


#### GAN model
<div align="center">
<img src=Figs/fig5.png>
</div>
The two spectral reconstruction methods use the GAN model, and their discriminators are both PatchGAN. (a) SRCGAN takes Conditional GAN as the main framework. (b) SAGAN includes SAP-UNet withoutboundary supervision and SAP-WNet with boundary supervision.

#### Dense Network
<div align="center">
<img src=Figs/fig6.png>
</div>
Spectral  reconstruction  methods  based  on  Dense  Network.  (a)  SRTiramisuNet.  (b)  HSCNN+,respectively HSCNN-U, HSCNN-R, and HSCNN-D from top to bottom.
 
Residual Network
<div align="center">
<img src=Figs/fig7.png>
</div>
Two spectral reconstruction methods based onResidual  Network.(a)  SREfficientNet.  (b)  SREffi-cientNet+ where CM, SM, and GM refer to Con-ditional Model, Specialized Model and Generic Modelrespectively.  

#### Attention Network
###### 1. SRAWAN
<div align="center">
<img src=Figs/fig8.png>
</div>
Adaptive  Weighted  Attention  Network  withcamera spectral sensitivity prior.  

###### 2. SRHRNet
<div align="center">
<img src=Figs/fig9.png>
</div>
4-level  Hierarchical  Regression  Network.  

##### 3. SRRPAN
<div align="center">
<img src=Figs/fig10.png>
</div>
Residual Pixel Attention Network.

##### Multi-branch Network
###### 1. SRLWRDNet
<div align="center">
<img src=Figs/fig11.png>
</div>
Light  Weight  Residual  Dense  Attention  Network.  

###### 2. SRPFMNet
<div align="center">
<img src=Figs/fig12.png>
</div>
Pixel-aware  Deep  Function-mixture  Network.  

## Results
<div align="center">
<img src=Figs/fig13.png>
</div>
Reconstruction accuracy comparison in terms of RMSE and MRAE on two selected datasets. The BGU-HS  and  ARAD-HS  datasets  are  proposed  by  pectral  reconstruction  challenges  NTIRE-2018  and  NTIRE-2020,respectively. The first and second best results are shown inboldand underlinerespectively.  
  
  

<div align="center">
<img src=Figs/fig14.png>
</div>
Generalization tests of three typical methods by changing image brightness in the ARAD-HS dataset.The minimum and maximum loss are shown inboldand underlinerespectively.


## References
### Papers - Prior-based methods
- Arad, Boaz, and Ohad Ben-Shahar. "Sparse recovery of hyperspectral signal from natural RGB images." In ECCV, 2016.[[Paper](https://link.springer.com/chapter/10.1007/978-3-319-46478-7_2)]
- Aeschbacher, Jonas, Jiqing Wu, and Radu Timofte. "In defense of shallow learned spectral reconstruction from rgb images." In ICCVW, 2017.[[Paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w9/Aeschbacher_In_Defense_of_ICCV_2017_paper.pdf)][[code](https://people.ee.ethz.ch/~timofter/)]
- Fu, Ying, et al. "Spectral reflectance recovery from a single rgb image." IEEE Transactions on Computational Imaging, 2018.[[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8410422)]
- Li, Yuqi, Chong Wang, and Jieyu Zhao. "Locally Linear Embedded Sparse Coding for Spectral Reconstruction From RGB Images." IEEE Signal Processing Letters, 2017.[[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8116687)]
- Geng, Yunhao, et al. "Spatial Constrained Hyperspectral Reconstruction from RGB Inputs Using Dictionary Representation." IGARSS ,2019.[[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8898871)]
- Jia, Yan, et al. "From RGB to spectrum for natural scenes via manifold-based mapping." In ICCV, 2017.[[paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Jia_From_RGB_to_ICCV_2017_paper.pdf)]
- Akhtar, Naveed, and Ajmal Mian. "Hyperspectral recovery from rgb images using gaussian processes." IEEE transactions on pattern analysis and machine intelligence, 2018.[[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8481553)]
### Papers - Data-driven methods
- Xiong, Zhiwei, et al. "Hscnn: Cnn-based hyperspectral image recovery from spectrally undersampled projections." In ICCVW, 2017.[[paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w9/Xiong_HSCNN_CNN-Based_Hyperspectral_ICCV_2017_paper.pdf)]
- Koundinya, Sriharsha, et al. "2d-3d cnn based architectures for spectral reconstruction from rgb images." In ICCVW, 2018.[[paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Koundinya_2D-3D_CNN_Based_CVPR_2018_paper.pdf)]
- Han, Xian-Hua, Boxin Shi, and Yinqiang Zheng. "Residual hsrcnn: Residual hyper-spectral reconstruction cnn from an rgb image." In ICPR, 2018.[[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8545634)]
- Stiebel, Tarek, et al. "Reconstructing spectral images from rgb-images using a convolutional neural network."In ICCVW, 2018.[[paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Stiebel_Reconstructing_Spectral_Images_CVPR_2018_paper.pdf)]
- Yan, Yiqi, et al. "Accurate spectral super-resolution from single RGB image using multi-scale CNN." In PRCV,  2018.[[paper](https://link.springer.com/chapter/10.1007/978-3-030-03335-4_18)][[code](https://github.com/ml-lab/Multiscale-Super-Spectral)]
- Banerjee, Atmadeep, and Akash Palrecha. "Mxr-u-nets for real time hyperspectral reconstruction." arXiv, 2020.[[paper](https://arxiv.org/pdf/2004.07003.pdf)][[code](https://github.com/akashpalrecha/hyperspectral-reconstruction)]
- Fubara, Biebele Joslyn, Mohamed Sedky, and David Dyke. "Rgb to spectral reconstruction via learned basis functions and weights." In CVPRW, 2020.[[paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Fubara_RGB_to_Spectral_Reconstruction_via_Learned_Basis_Functions_and_Weights_CVPRW_2020_paper.pdf)]
- Alvarez-Gila, Aitor, Joost Van De Weijer, and Estibaliz Garrote. "Adversarial networks for spatial context-aware spectral image reconstruction from rgb." In CVPRW, 2018.[[paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w9/Alvarez-Gila_Adversarial_Networks_for_ICCV_2017_paper.pdf)]
- Liu, Pengfei, and Huaici Zhao. "Adversarial Networks for Scale Feature-Attention Spectral Image Reconstruction from a Single RGB." Sensors, 2020.[[paper](https://www.mdpi.com/1424-8220/20/8/2426)]
- Galliani, Silvano, et al. "Learned spectral super-resolution." arXiv ,2017.[[paper](https://arxiv.org/pdf/1703.09470.pdf)]
- Shi, Zhan, et al. "Hscnn+: Advanced cnn-based hyperspectral recovery from rgb images." IN CVPRW, 2018.[[paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Shi_HSCNN_Advanced_CNN-Based_CVPR_2018_paper.pdf)][[code](https://github.com/ngchc/HSCNN-Plus)]
- Can, Yigit Baran, and Radu Timofte. "An efficient CNN for spectral reconstruction from RGB images." arXiv, 2018.[[paper](https://arxiv.org/pdf/1804.04647.pdf)][[code](https://github.com/ybarancan/efficient_spectral_cnn)]
- Kaya, Berk, Yigit Baran Can, and Radu Timofte. "Towards spectral estimation from a single RGB image in the wild." In ICCVW, 2019.[[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9022323)][[code](https://github.com/berk95kaya/Spectral-Estimation)]
- Li, Jiaojiao, et al. "Adaptive weighted attention network with camera spectral sensitivity prior for spectral reconstruction from RGB images." In CVPRW, 2020.[[paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Li_Adaptive_Weighted_Attention_Network_With_Camera_Spectral_Sensitivity_Prior_for_CVPRW_2020_paper.pdf)][[code](https://github.com/Deep-imagelab/AWAN)]
- Zhao, Yuzhi, et al. "Hierarchical regression network for spectral reconstruction from RGB images." In CVPRW, 2020.[[paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Zhao_Hierarchical_Regression_Network_for_Spectral_Reconstruction_From_RGB_Images_CVPRW_2020_paper.pdf)][[code](https://github.com/zhaoyuzhi/Hierarchical-Regression-Network-for-Spectral-Reconstruction-from-RGB-Images)]
- Peng, Hao, Xiaomei Chen, and Jie Zhao. "Residual pixel attention network for spectral reconstruction from rgb images." In CVPRW, 2020.[[paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Peng_Residual_Pixel_Attention_Network_for_Spectral_Reconstruction_From_RGB_Images_CVPRW_2020_paper.pdf)]
- Nathan, D. Sabari, et al. "Light weight residual dense attention net for spectral reconstruction from rgb images." arXiv, 2020.[[paper](https://arxiv.org/ftp/arxiv/papers/2004/2004.06930.pdf)]
- Zhang, Lei, et al. "Pixel-aware deep function-mixture network for spectral super-resolution."  In AAAI, 2020.[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6978)]

