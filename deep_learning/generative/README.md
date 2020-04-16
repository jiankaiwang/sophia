# Generative Adversarial Network, GAN

Generative Adversarial Network (GAN) is composed of two main subnetworks, a `generative` one and a `discriminative` one. They are against each other. A generative submodel tries to generate similar content with the training dataset and the other discriminative one tries to distinguish the generated content coming from the real one or the fake one.

This is a good idea of a balance between two against forces. The generative model tries to find the keys of features describing the dataset and the discriminator one tries to find the keys of features distinguishing from the differences.

## Content

* Vanilla GAN : [ipynb](SimpleGAN_Keras.ipynb)
* Multiple Generator GAN : [ipynb](MultiGenerator_GAN_Keras.ipynb)

## Others Related Repositories

* DCGAN (Deep Convolutional GAN)
    * https://github.com/jacobgil/keras-dcgan
* CycleGAN (image-to-image translation)
    * https://github.com/xhujoy/CycleGAN-tensorflow
* SSGAN (Semi-supervised Learning GAN)
    * https://github.com/gitlimlab/SSGAN-Tensorflow