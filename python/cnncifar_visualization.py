# -*- coding: utf-8 -*-

"""
Conv-Deconvolutional Network.
"""

__authors__ = "Nicolas Laliberte, Jimmy Leroux"
__version__ = "1.2"
__maintainer__ = "Nicolas Laliberté"

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time

class cov_net(nn.Module):
	"""
	Convolutional network
	"""
	def __init__(self):
		super(cov_net, self).__init__()
		"""
		Indices representing the position in the network for conv, relu and
		max pool layers.
		"""
		self.conv_indices = []
		self.relu_indices = []
		self.maxp_indices = []
		
		"""
		Network to be read in order.
		"""
		self.layers = nn.Sequential(
			#Conv 1
			nn.Conv2d(3, 64, 3, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(64),
			nn.Dropout2d(p=0.05),
			
			#Conv2
			nn.Conv2d(64, 128, 3, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(128),
			nn.Dropout2d(p=0.05),
			
			#Conv3
			nn.Conv2d(128, 128, 3, padding=1), 
			nn.ReLU(),
			nn.BatchNorm2d(128),
			nn.MaxPool2d(2, stride = 2, return_indices = True),
			nn.Dropout2d(p=0.05),
			
			#Conv4
			nn.Conv2d(128, 256, 3, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(256),
			nn.Dropout2d(p=0.05),
			
			#Conv5
			nn.Conv2d(256, 256, 3, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(256),
			nn.MaxPool2d(2, stride = 2, return_indices = True),
			nn.Dropout2d(p=0.05),
			
			#Conv6
			nn.Conv2d(256, 512, 3, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(512),
			nn.Dropout2d(p=0.05),
			
			#Conv7
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(2, stride = 2, return_indices = True),
			nn.Dropout2d(p=0.05),
			
			#Conv8
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(512),
			nn.Dropout2d(p=0.05))

		#We keep the output of each layer for the deconv net.
		self.layers_outputs = [0]*len(self.layers)
		#We keep the max pool indices to compute the partial inverse in the
		#deconv net.
		self.pool_indices = {}
		
		"""
		Classifier at the end of the CNN.
		"""
		self.classifier = nn.Sequential(
			nn.Linear(512 * 4 * 4, 1024),
			nn.ReLU(),
			nn.BatchNorm1d(1024),
			nn.Dropout2d(p=0.20),
			nn.Linear(1024, 10))
		
		self.init_indices()

	def init_indices(self):
		for i, layer in enumerate(self.layers):
			if isinstance(layer, nn.MaxPool2d):
				self.maxp_indices.append(i)
			elif isinstance(layer, nn.ReLU):
				self.relu_indices.append(i)
			elif isinstance(layer, nn.Conv2d):
				self.conv_indices.append(i)
			 
	def forward_features(self, x):
		"""
		Inputs:
		-------------------------------
		x: Images with 3 channels.
		
		Returns: 
		-------------------------------
		Output obtained by forwarding the input through the CNN.     
		
		"""
		output = x
		for i, layer in enumerate(self.layers):
			if isinstance(layer, nn.MaxPool2d):
				output, indices = layer(output)
				self.layers_outputs[i] = output
				self.pool_indices[i] = indices
			else:
				output = layer(output)
				self.layers_outputs[i] = output
		return output
	
	def eval_features(self, x):
		"""
		Inputs: 
		-------------------------------
		x: Images with 3 channels.
		
		Returns: 
		-------------------------------            
		Output obtained by forwarding the input through the CNN without dropout
		and batch norm. This function is used to extract feature with the 
		deconv net.        
		"""
		output = x
		for i, layer in enumerate(self.layers):
			if isinstance(layer, nn.MaxPool2d):
				output, indices = layer(output)
				self.layers_outputs[i] = output
				self.pool_indices[i] = indices
			elif isinstance(layer, nn.Conv2d):
				output = layer(output)
				self.layers_outputs[i] = output
			elif isinstance(layer, nn.ReLU):
				output = layer(output)
				self.layers_outputs[i] = output
		return output
	
	def forward(self, x):
		output = self.forward_features(x)
		output = output.view(output.size()[0], -1)
		output = self.classifier(output)
		return output

class deconv_net(nn.Module):
	"""
	Deconvolutional network.
	"""
	def __init__(self, cnn):
		super(deconv_net, self).__init__()
		self.cnn = cnn
		"""
		Dictionaries of indices to associate the layers from the CNN to the 
		deconv net. As an exemple: conv2DeconvIdx[0] means that the 17th layer
		is 'attached' to the first layer of CNN.
		"""
		self.conv2DeconvIdx = {}
		self.relu2DeconvIdx = {}
		self.maxp2DeconvIdx = {}
		
		"""
		Indices needed to initialize weight and for max unpool.
		"""
		self.biasIdx = {}
		self.unpool2Idx = {}
		
		"""
		Deconvolutional neural network associated with a given cnn.
		"""
		self.deconv_features = nn.ModuleList([inverse_module(i) 
			for i in reversed(cnn.features) if inverse_module(i) is not None])
				
		"""
		We choose only one feature map to reconstruct the image. So we need 
			for the first step to pass through a (1 , N) size ConvTranspose2d.
		"""
		self.deconv_max_filter = nn.ModuleList([inverse_module(i, True) 
			for i in reversed(cnn.features) if inverse_module(i) is not None])
		
		self.init_indices()
		self.initialize_weights()

		
	def init_indices(self):
		"""
		Initialize dictionary of indices associating each deconv layer with the
		proper conv layer.
		"""
		idx_conv = 0
		idx_relu = 0
		idx_maxp = 0
		for i, layer in enumerate(self.deconv_features):
			if isinstance(layer, nn.ConvTranspose2d):
				self.conv2DeconvIdx[cnn.conv_indices[-1 - idx_conv]] = i
				if i <= 1:
					self.biasIdx[cnn.conv_indices[-1 - idx_conv]] = 0
				else:
					self.biasIdx[cnn.conv_indices[-1 - idx_conv]] = \
						self.conv2DeconvIdx[cnn.conv_indices[-idx_conv]]
				idx_conv += 1
			if isinstance(layer, nn.ReLU):
				self.relu2DeconvIdx[cnn.relu_indices[-1 - idx_relu]] = i
				idx_relu += 1
			if isinstance(layer, nn.MaxUnpool2d):
				self.maxp2DeconvIdx[cnn.maxp_indices[-1 - idx_maxp]] = i
				self.unpool2Idx[i] = cnn.maxp_indices[-1 - idx_maxp]
				idx_maxp += 1
	
	def initialize_weights(self):
		"""
		Input:
		-------------------------------
		cnn : Pre-trained cnn.
		
		-------------------------------
		Set weights and bias of the deconvulational layer to those of the input
		cnn.
		"""
		for i, layer in enumerate(self.cnn.features):
			if isinstance(layer, nn.Conv2d):
				self.deconv_features[self.conv2DeconvIdx[i]].weight.data = \
					layer.weight.data
				biasIdx = self.biasIdx[i]
				if biasIdx > 0:
					self.deconv_features[biasIdx].bias.data = \
						layer.bias.data.to(device)
		self.deconv_features[len(self.deconv_features) - 1].bias.data = \
			self.deconv_features[len(self.deconv_features) - 1].bias.data.to(device)              
				
	def forward(self, x, layer, filt_number, pool_indices):
		"""
		Inputs:
		-------------------------------
		x: Feature map to reconstruct.
		layer: Layer from which we reconstruct the feature map.
			
			
		"""
		
		#Depending at which state in the cnn we decided to reconstruct the image.
		#There are 3 possibilities: Conv2d, ReLU, MaxPool. 
		if isinstance(self.cnn.features[layer], nn.MaxPool2d):
			start_idx = self.maxp2DeconvIdx[layer]
			output = self.deconv_max_filter[start_idx](x, 
				pool_indices[self.unpool2Idx[start_idx]])
			start_idx += 1
			output = self.deconv_max_filter[start_idx](output)
			start_idx += 1
			idmax, M = max_filter(output)
			fsize = output.size()[2]
			output = output[0][idmax].view(1,1,fsize,fsize)           
			
		elif isinstance(self.cnn.features[layer], nn.ReLU):
			start_idx = self.relu2DeconvIdx[layer]
			output = self.deconv_max_filter[start_idx](x)
			start_idx += 1
			
		elif isinstance(self.cnn.features[layer], nn.Conv2d):
			start_idx = self.conv2DeconvIdx[layer]
			output = x
			
		self.deconv_max_filter[start_idx].weight.data = \
			self.deconv_features[start_idx].weight[filt_number].data[None, 
			:, :, :]
		 
		self.deconv_max_filter[start_idx].bias.data = \
			self.deconv_features[start_idx].bias.data
		
		output = self.deconv_max_filter[start_idx](output)
		for i in range(start_idx + 1, len(self.deconv_features)):
			if isinstance(self.deconv_features[i], nn.MaxUnpool2d):
				output = self.deconv_features[i](output, 
					pool_indices[self.unpool2Idx[i]])
			else:
				output = self.deconv_features[i](output)
		return output
	
def inverse_module(layer, uni_filter = False):
	"""
	Inputs:
	-------------------------------
	layer: Layer to be inverse in order to create the deconv net. This function
	takes only the ReLU, MaxPool2d or ConvTranspose2d module as inputs.
	uni_filter: Boolean input (Default = False). This input is for conv2d 
	layer, if True it will return ConvTranspose2d layer of size (1, x) in order
	to reconstruct an image for one specific feature map.
	
	Returns:
	-------------------------------
	If ReLU, MaxPool2d or ConvTranspose2d is given, then it returns ReLU,
	MaxUnpool2d or ConvTranspose2d respectively. If not, it returns None.
	
	"""
	if isinstance(layer, nn.ReLU):
		return nn.ReLU()
	if isinstance(layer, nn.MaxPool2d):
		return nn.MaxUnpool2d(2, stride = 2)
	if isinstance(layer, nn.Conv2d):
		input_size = layer.state_dict()['weight'].size()[1]
		output_size = layer.state_dict()['weight'].size()[0]
		kernel_size = layer.state_dict()['weight'].size()[2]
		if not uni_filter:
			return nn.ConvTranspose2d(output_size, input_size, kernel_size, 
				padding = 1)
		if uni_filter:
			return nn.ConvTranspose2d(1, input_size, kernel_size, padding = 1)
	else:
		return None
	
	
def reconstruction(image, cnn, layer):
	"""
	Inputs: 
	-------------------------------
	image: Images from the data set.
	cnn: Pre-trained CNN from which we want to visualize learned features.
	layer: From which layer we want to recontruct to image.    
	
	Returns:
	-------------------------------
	Reconstruction of 2 images: one from a randomly choosen filter at the input
	layer and from the max filter of the same layer.
	"""
	deconv = deconv_net(cnn)
	cnn.eval_features(image.view(1,3,32,32).to(device))
	feat = cnn.features_outputs[layer]
	idmax, M = max_filter(feat)
	poolIdx = cnn.pool_indices
	
	if isinstance(cnn.features[layer], nn.MaxPool2d):
		recont_max = deconv.forward(feat, layer, idmax, poolIdx)
		recont_max = recont_max.detach().to("cpu").numpy()[0]
		recont_max = np.transpose(recont_max, (1,2,0))
		
		return recont_max, recont_max
	else:
		fsize = feat.size()[2]
	
		recont_max = deconv.forward(feat[0][idmax].view(1,1,fsize,fsize),
									layer, idmax, poolIdx)
		recont_max = recont_max.detach().to("cpu").numpy()[0]
		recont_max = np.transpose(recont_max, (1,2,0))
	
		recont_rand = deconv.forward(feat[0][0].view(1,1,fsize,fsize),
									layer, 0, poolIdx)
		recont_rand = recont_rand.detach().to("cpu").numpy()[0]
		recont_rand = np.transpose(recont_rand, (1,2,0))
	
		return recont_max, recont_rand

def max_filter(filters):
	"""
	Inputs:
	-------------------------------
	Tensor of size (N, (filters))
	
	Returns:
	-------------------------------
	The max based on means and std from the N filters and the positions of this
	filter.
	"""
	idmax = 0
	M = 0.
	for i in range(len(filters)):
		measure = torch.sum(filters[0][i])*torch.std(filters[0][i])
		if measure > M:
			idmax = i
			M = measure
	return idmax, M 

def eval(cnn, testloader):
	"""
	Function evaluating the performance of the model (cnn) on the data (
	testloader).

	Inputs:
	-------
	cnn: cov_net instance we want to test the performance.
	testloader: Pytorch Dataloader containing the testing data.

	Prints:
	-------
	Compute the performances on each of the 10 classes and plot the associated
	confusion matrix.
	"""

	correct = 0
	total = 0
	cnn.eval()
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = cnn(images.to(device))
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels.to(device)).sum().item()

	print('Accuracy of the network on the 1000 test images: %d %%' % (
		100 * correct / total))

	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	confusion = torch.zeros(10,10)
	count = 0  
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = cnn(images.to(device))
			_, predicted = torch.max(outputs, 1)
			c = (predicted == labels.to(device)).squeeze()
			for i in range(c.shape[0]):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1
				confusion[label,predicted[i].item()] += 1
				count += 1
		plt.imshow(confusion/torch.tensor(class_total).view(10,1))
		plt.colorbar()
		plt.yticks(range(10), classes)
		plt.xticks(range(10), classes, rotation='vertical')
		plt.xlabel('Predicted')
		plt.ylabel('True')
	for i in range(10):
		if class_total[i]!=0:
			print('Accuracy of %5s : %2d %%' % (
				classes[i], 100 * class_correct[i]/class_total[i]))
		else:
			print('Accuracy of %5s : %2d %%' % (
				classes[i], 100 * class_correct[i]))

def train(cnn, trainloader, testloader, num_epoch=20, lr=0.01,
		weight_decay=0.0001):
	"""
	Train function for the network.

	Inputs:
	-------
	cnn: conv_net instance we want to train.
	trainloader: pytorch Dataloader instance containing the training data.
	testloader: pytorch Dataloader instance containing the test data.
	num_epoch: number of training epoch.
	lr: Learning rate for the SGD.
	weight_decay: value of the L2 regularization.

	Returns:
	-------
	loss_train: normalized loss on the training data at after each epoch.
	loss_test: normalized loss on the test data at after each epoch.
	err_train: total error on the training set after each epoch.
	err_test: total error on the test set after each epoch.
	"""    
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(cnn.parameters(), lr=lr, weight_decay=weight_decay,
		momentum=0.9)
	
	loss_train = []
	loss_test = []
	err_train = []
	err_test = []
	for epoch in range(num_epoch):

		correct = 0.
		total = 0.
		running_loss_train = 0.0
		running_loss_test = 0.0
		cnn.train()
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			optimizer.zero_grad()
			outputs = cnn(inputs.to(device))
			loss = criterion(outputs, labels.to(device))
			loss.backward()
			optimizer.step()
			running_loss_train += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels.to(device)).sum().item()
		err_train.append(1 - correct / total)
		
		correct = 0.
		total = 0.
		cnn.eval()
		with torch.no_grad():
			for data in testloader:
				images, labels = data
				outputs = cnn(images.to(device))
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels.to(device)).sum().item()
				loss = criterion(outputs, labels.to(device))
				running_loss_test += loss.item()
		err_test.append(1 - correct / total)
		
		loss_train.append(running_loss_train)
		loss_test.append(running_loss_test)
		print('Epoch: {}'.format(epoch))
		print('Train loss: {0:.4f} Train error: {1:.2f}'.format(
			loss_train[epoch], err_train[epoch]))
		print('Test loss: {0:.4f} Test error: {1:.2f}'.format(
		   loss_test[epoch], err_test[epoch]))

	print('Finished Training')
	return loss_train, loss_test, err_train, err_test

def visualization(i, layer, gray = True):
	"""
	Inputs:
	-------------------------------
	i: Indices in the dataset of images.
	layer: From which layer we want to visualize features.
	gray: True if we want to visualize in a 'electrical' view. False if not.
	
	Outputs:
	-------------------------------
	Recontruction of the image.
	"""    
	recont_max, recont_rand = reconstruction(trainset[i][0], cnn, layer)
	
	if gray:
		gray_feat = np.sum(recont_max, axis = 2)/3
		return gray_feat

	else:
		recont_max = (recont_max - recont_max.min())/(recont_max.max() - 
					  recont_max.min())*255
		feat_max = recont_max.astype(np.uint8)
		return feat_max
	
def show_image(i):
	"""
	Inputs:
	-------------------------------
	i: Indices in the dataset of images.
	
	Returns:
	-------------------------------
	Image renormalized and ready to be used in plt.imshow.
	"""
	im_init = trainset[i][0].detach().to("cpu").numpy()
	im_init = np.transpose(im_init, (1,2,0))
	im_init = im_init/2 + 0.5
	
	return im_init
			
def grid_feature(indices, layers = [0,4,27]):
	"""
	Inputs:
	-------------------------------
	indices: List of indices in the dataset of images from which we want to
			 visualize learned features from the cnn.
	layers: List of layers from which we want to reconstruct the images.
	
	Returns:
	-------------------------------
	Show a grid displaying all the features and choosen images.
	"""
	xn = len(indices)
	yn = len(layers) + 1
	f, grid_im = plt.subplots(xn, yn, sharex = True)
	f.subplots_adjust(left = 0.3, right = 0.8, hspace = 0.1, wspace = 0.1)
	x = 0
	for i in indices:
		y = 0
		for l in layers:
			vis = visualization(i, l)
			grid_im[x, y].imshow(vis, cmap = 'bone')
			grid_im[x, y].get_xaxis().set_visible(False)
			grid_im[x, y].get_yaxis().set_visible(False)            
			y += 1
		im = show_image(i)
		grid_im[x, yn - 1].imshow(im)
		grid_im[x, yn - 1].get_xaxis().set_visible(False)
		grid_im[x, yn - 1].get_yaxis().set_visible(False)
		x += 1
	
	
	grid_im[0,0].set_title("First layer", fontsize = 9)
	grid_im[0,1].set_title("Second layer", fontsize = 9)
	grid_im[0,2].set_title("Last layer", fontsize = 9)
	grid_im[0,3].set_title("Images", fontsize = 9)
	plt.show()

def find_label(labels, start = 0):
	"""
	Function that finds the first indice encountered from the starting point
	in the dataset that represents the class in the labels inputs.
	
	Inputs:
	-------------------------------
	labels: List of int representing the classes we are looking for.
	start: Starting indice in the dataset.
	
	Returns:
	-------------------------------
	List of indices in the dataset.
	"""
	indices = []
	for i in range(start, len(trainset)):
		if len(labels) == 0:
			break
		label = trainset[i][1]
		if label in labels:
			ind = labels.index(label)
			labels.pop(ind)
			indices.append(i)
	return indices               

if __name__ == '__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	torch.cuda.manual_seed(10)
	plt.style.use('ggplot')     
	plt.rc('xtick', labelsize=15)
	plt.rc('ytick', labelsize=15)
	plt.rc('axes', labelsize=15)

	transform = transforms.Compose([transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])      

	transform_t = transforms.Compose(
		[transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),transforms.RandomApply(
		[transforms.RandomRotation(10)],0.0),transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

   
	
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
										download=True, transform=transform_t)
	
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
											  shuffle=True, num_workers=0)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
										download=True, transform=transform_t)
	
	testloader = torch.utils.data.DataLoader(testset, batch_size=64,
											 shuffle=False, num_workers=0)    

	classes = ('plane', 'car', 'bird', 'cat',
		   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	cnn = cov_net().to(device)
	
	t1 = time.time()
	num_epoch = 1
	loss_train, loss_test, e_train, e_test = train(cnn, trainloader,
		testloader, num_epoch) 
	print(time.time()-t1)
	
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
								download=True, transform=transform)

	eval(cnn, testloader)   
	
	plt.figure()
	plt.plot(range(1, num_epoch + 1),loss_train, 'sk-', label='Train')
	plt.plot(range(1, num_epoch + 1),loss_test, 'sr-', label='Valid')
	plt.xlabel('Epoch')
	plt.ylabel('Average cost')
	plt.legend()

	plt.figure()
	plt.plot(range(1, num_epoch + 1),e_train, 'sk-', label='Train')
	plt.plot(range(1, num_epoch + 1),e_test, 'sr-', label='Valid')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.legend()
	
	labels = find_label([3, 5, 7, 9])
	grid_feature(labels)
	
	torch.cuda.empty_cache()
	plt.show()

