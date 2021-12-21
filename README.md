# Accelerated Randomized Methods for Receiver Design in Extra-Large Scale MIMO Arrays
This is a research-oriented code package that is primarily intended to allow readers to replicate the results of the article mentioned below and also encourage and accelerate further research on this topic:

V. Croisfelt, A. Amiri, T. Abrao, E. D. Carvalho and P. Popovski, "Accelerated Randomized Methods for Receiver Design in Extra-Large Scale MIMO Arrays," in IEEE Transactions on Vehicular Technology, doi: 10.1109/TVT.2021.3082520. Available on: https://ieeexplore.ieee.org/document/9437708.

The package is based on the Python language and can, in fact, reproduce all the numerical results discussed in the article. To contextualize, in the sequel, we present the abstract of the article and other important information.

I hope this content helps in your reaseach and contributes to building the precepts behind open science. Remarkably, in order to boost the idea of open science and further drive the evolution of science, we also motivate you to share your published results to the public.

If you have any questions and if you have encountered any inconsistency, please do not hesitate to contact me via victorcroisfelt@gmail.com.

## Abstract
Massive multiple-input-multiple-output (M-MIMO) features a capability for spatial multiplexing of large number of users. This number becomes even more extreme in extra- large (XL-MIMO), a variant of M-MIMO where the antenna array is of very large size. Yet, the problem of signal processing complexity in M-MIMO is further exacerbated by the XL size of the array. The basic processing problem boils down to a sparse system of linear equations that can be addressed by the randomized Kaczmarz (RK) algorithm. This algorithm has re- cently been applied to devise low-complexity M-MIMO receivers; however, it is limited by the fact that certain configurations of the linear equations may significantly deteriorate the performance of the RK algorithm. In this paper, we embrace the interest in accelerated RK algorithms and introduce three new RK-based low-complexity receiver designs. In our experiments, our methods are not only able to overcome the previous scheme, but they are more robust against inter-user interference (IUI) and sparse channel matrices arising in the XL-MIMO regime. In addition, we show that the RK-based schemes use a mechanism similar to that used by successive interference cancellation (SIC) receivers to approximate the regularized zero-forcing (RZF) scheme.

## Content
The codes provided here can be used to simulate Figs. 3 to 6. The data of each figure is obtained by executing the script named by the respective figure. Further details about each file can be found inside them.

## Citing this Repository and License
This code is subject to the MIT license. If you use any part of this repository for research, please consider to cite our aforementioned work.

## Acknowledgments
This research was supported in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) under grant 88887.461434/2019-00.
