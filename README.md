# Network architecture
a
# Cifar-10 visualization
We present here an implementation of a Convolutional Neural Network (CNN) on the Cifar-10 dataset for feature visualization. For the visualization, we propagate an image through the trained network then select one of the feature map which we project down to the pixel space with a deconvolution network. See the image below for an exemple of the deconvoluted feature maps.

![vis](https://github.com/jimleroux/Cifar-10-visualization/blob/master/png/visualization.png)

We chose the maps that had the maximum activation and the maximum variance.
