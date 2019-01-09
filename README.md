# Cifar-10 visualization
We present here an implementation of a Convolutional Neural Network (CNN) on the Cifar-10 dataset for feature visualization. The code is provided in the python directory.
## Network architecture
The network architecture we used is a very basic multilayed CNN, see the figure below.
<img src="https://github.com/jimleroux/Cifar-10-visualization/blob/master/png/architecture.png" height="150" width="600> 
![cnn](https://github.com/jimleroux/Cifar-10-visualization/blob/master/png/architecture.png)

We start with the inputs images (In) and feed them in the network. The convulutional layers are indicated by the letters C with the number of filters indicated below. 3x3 filters with a padding = 1 and stride = 1 was used. The layers marked by S are subsambling layers. They are composed of a 2x2 Maxpooling followed by a convolutional layer. On top of each convulutional layers, we have a dropout layer (5\%) and a batchnorm layer. At the end on the network, there are two fully connected layers (FC) used to make the predictions. The first (FC) contains a dropout layer (20\%) and a batchnorm layer.
## Deconvolution network
For the visualization, we propagate an image through the trained network then select one of the feature map which we project down to the pixel space with a deconvolution network. See the image below for an exemple of the deconvoluted feature maps.

![vis](https://github.com/jimleroux/Cifar-10-visualization/blob/master/png/visualization.png)

We chose the maps that had the maximum activation and the maximum variance.
