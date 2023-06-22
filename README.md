A New Paradigm for Generative Adversarial Networks based on Randomized Decision Rules
===============

The code includes the experiments of the Empirical Bayesian GAN(EBGAN) by Sehwan Kim, Qifan Song, and Faming Liang. We propose to train the GAN by an empirical Bayes-like method by treating the discriminator as a hyper-parameter of the posterior distribution of the generator. Specifically, we simulate generators from its posterior distribution conditioned on the discriminator using a stochastic gradient Markov chain Monte Carlo (MCMC) algorithm, and update the discriminator using stochastic gradient descent along with simulations of the generators. 

## Related Publication

Sehwan Kim, Qifan Song, and Faming Liang (2023+), A New Paradigm for Generative Adversarial Networks based on Randomized Decision Rules, accepted by *Statistica Sinica*

## Description

EBGAN uses multiple generators and one discriminator to address the mode collapse issue. As introduction of mode collapse issue for vanilla GAN, we considered the one mode gaussian example. Below is the example for  

Key properties of the Bayesian approach to GANs include (1) accurate predictions on semi-supervised learning problems; (2) minimal intervention for good performance; (3) a probabilistic formulation for inference in response to adversarial feedback; (4) avoidance of mode collapse; and (5) a representation of multiple complementary generative and discriminative models for data, forming a probabilistic ensemble.

<p align="center">
    <img src="img/download (10).png" width="500"/>
</p>

You can Refer the notebook `.ipynb`.

