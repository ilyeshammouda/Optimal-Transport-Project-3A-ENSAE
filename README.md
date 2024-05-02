# Optimal-Transport-Project-3A-ENSAE
<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="ENSAE.png" alt="Project logo"></a>
</p>

<h3 align="center">A Specialized Semismooth Newton Method for Kernel-Based
Optimal Transport</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---



## 📝 Table of Contents

- [About](#about)
- [Installing](#Installing)
- [Results](#Results)
- [Conclusion](#Conclusion)
- [Authors](#authors)
- [References](#References)

##  About <a name = "about"></a>
This project describes an implementation of semismooth Newton (SSN) method for kernel-based optimal transport (OT).The aim of this project is to
implement a Specialized Semismooth Newton Method (SSN) for Kernel-Based Optimal Transport,
which was introduced by Lin et al. [2024]. After initially implementing the algorithm primarily using numpy, we attempted to re-implement
it in JAX.




## Installing <a name = "Installing"></a>

To ensure you have all the libraries used in our simulations, refer to the [requirements.txt](requirements.txt) file. Use the following command to install any missing libraries:

```
pip install -r requirements.txt
```




##  Running the tests <a name = "tests"></a>

First, execute the [preprocessing](preprocessing.py) script to reorganize your data into a different folder. After adjusting the dataset path and fixing the output path, run the preprocessing using the following command:

```bash
python3 preprocessing.py
```

Now you are prepared to replicate the experiments using our two notebooks. Start by running [GAN.ipynb](GAN.ipynb) to generate poisoned images. Subsequently, use [Classifier.ipynb](Classifiers.ipynb) to conduct experiments.


## Results <a name="Results"></a>

### Examples of Produced Perturbations

Here are a few instances of perturbation as we alter the parameter $\alpha$, which signifies the intensity of L2 penalization.

<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">

  <div style="flex-basis: 20%;">
    <img src="images/alpha1.png" alt="Image with $\alpha$ = 1" width="10%">
    <p style="text-align: center;">Figure 1: $\alpha$ = 1</p>
  </div>

  <div style="flex-basis: 20%;">
    <img src="images/alpha100.png" alt="Image with $\alpha$ = 100" width="10%">
    <p style="text-align: center;">Figure 2: $\alpha$ = 100</p>
  </div>

  <div style="flex-basis: 20%;">
    <img src="images/alpha1000.png" alt="Image with $\alpha$ = 1000" width="10%">
    <p style="text-align: center;">Figure 3: $\alpha$ = 1000</p>
  </div>

</div>


Here are a few instances of perturbation as we modify the number of encoders, from 1 to 5.

<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">

  <div style="flex-basis: 20%;">
    <img src="images/1bkb.png" alt="Image with $\alpha$ = 1" width="20%">
    <p style="text-align: center;">Figure 1: 1 Backbone (ResNest26D)</p>
  </div>

  <div style="flex-basis: 20%;">
    <img src="images/3bkb.png" alt="Image with $\alpha$ = 100" width="20%">
    <p style="text-align: center;">Figure 2: 3 Backbones (ResNet34, EfficientNetB0, and Resnest26d)</p>
  </div>

  <div style="flex-basis: 20%;">
    <img src="images/5bkb.png" alt="Image with $\alpha$ = 1000" width="20%">
    <p style="text-align: center;">Figure 3: 5 Backbones (ResNet34, EfficientNetB0, Resnest26d, RegNetX_006, and DenseNet121)</p>
  </div>

</div>

With substantial L2 penalization and an increased number of encoders, the perturbation becomes less perceptible.

In a black-box context, where the target network differs from surrogates, we evaluate performance degradation across various backbones—utilizing 3 and 5 surrogate networks. For ResNest101e, akin to ResNest26D, accuracy results are detailed in the table below.

|                 | Clean | 10% Poisoned | 50% Poisoned | 100% Poisoned |
|-----------------|-------|--------------|--------------|---------------|
| 3 Backbones     | 0.5283| 0.6518       | 0.6214       | 0.1988        |
| 5 Backbones     | 0.5283| 0.6515       | 0.6329       | 0.2013        |

*Table 1: Accuracies obtained with ResNest101e, with either 3 or 5 backbones for the discriminator, with varying quantities of poisoned images.*

When the entire training set is poisoned (100%), accuracy decreases from 53% to 20%. Interestingly, when only a portion of the dataset is poisoned, the accuracy exceeds that of the clean dataset—a phenomenon we will delve into further in the Discussion section.

More results and technical details are presented and discussed in our [report](report.pdf).


## Conclusion <a name="Conclusion"></a>

In this project, we introduced an innovative adversarial approach aimed at compromising the effectiveness of a deep neural network classifier by subtly introducing contamination into the training dataset. Surrogate backbones were employed as evaluative entities to measure adaptability to alternative backbones, evaluating performance in both transparent and opaque scenarios.

Our experiments on the German Traffic Sign Recognition dataset successfully showcased the introduced contamination, resulting in a notable decline in classifier accuracy across both scenarios. Interestingly, incorporating a fraction of the contaminant unexpectedly led to an improvement in accuracy, challenging established assumptions and prompting a need for deeper investigation.

Ablation experiments confirmed the importance of integrating both perturbation and counterfeit detection components within the methodology. These unexpected findings suggest a complex interplay between adversarial elements and model performance, emphasizing the need for further investigation and potential applications in various computer vision tasks.







## ✍️ Authors <a name = "authors"></a>

- [@Valentin](https://github.com/Tordjx) 
- [@Ambre](https://github.com/ambree14) 
- [@Ilyes](https://github.com/ilyeshammouda) 



## References <a name = "References"></a>
- C.Zhu,W.R.Huang,A.Shafahi,H.Li,G.Taylor,C.Studer,T.Goldstein. [Transferable Clean-Label Poisoning Attacks on Deep Neural Nets](https://doi.org/10.48550/arXiv.1905.05897): [arXiv:1905.05897v2](https://doi.org/10.48550/arXiv.1905.05897)
- K.He, X.Zhang,S.Ren,J.Sun. [Deep Residual Learning for Image Recognition](https://doi.org/10.48550/arXiv.1512.03385): [	arXiv:1512.03385 ](
https://doi.org/10.48550/arXiv.1512.03385)
- Z.Zhou, M.R.Siddiquee,N.Tajbakhsh, J.Liang. [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://doi.org/10.48550/arXiv.1807.10165): [arXiv:1807.10165v1 ](
https://doi.org/10.48550/arXiv.1807.10165
)
