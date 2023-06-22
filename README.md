A New Paradigm for Generative Adversarial Networks based on Randomized Decision Rules
===============

The code includes the experiments of the EBGAN
We propose to train the GAN by an empirical Bayes-like method by treating the discriminator as a hyper-parameter of the posterior distribution of the generator. Specifically, we simulate generators from its posterior distribution conditioned on the discriminator using a stochastic gradient Markov chain Monte Carlo (MCMC) algorithm, and update the discriminator using stochastic gradient descent along with simulations of the generators. 

Sehwan Kim, Qifan Song, and Faming Liang (2023+), A New Paradigm for Generative Adversarial Networks based on Randomized Decision Rules, accepted by Statistica Sinica

