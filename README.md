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
- [Code description](#Codedescription)


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




##  Code description <a name = "Codedescription"></a>

The main code of this project is organized in two notebooks. The first one, 'python_numpy_implementation.ipynb', contains our implementation of the algorithm using classical techniques, for instance, numpy. While the second one, 'python_jax_implementation.ipynb', contains a way of implementing the same algorithm using JAX. In the file    you can find an implementation of kernel-based calculations for optimal transport, offering functionality for computing kernel matrices, distance matrices, and estimating transport costs. It supports both Sobolev and Gaussian kernels and provides methods for constructing kernels from samples and estimating optimal transport costs in one-dimensional settings. Overall, it serves as a toolkit for analyzing and solving optimal transport problems using kernel methods.







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
