from fastai.vision import *
from skimage import io

import matplotlib.pyplot as plt
import numpy as np
import warnings

import sys, os

def get_image_size(path):
	"""
	Returns image size of image at 'path' location
	:param path: path to target image
	"""
	imgEx = io.imread(path,as_gray=True)
	return imgEx.shape

def show_example_data_batch(image_path, img_df:DataFrame, get_labels:Callable, get_weights:Callable,cmap='viridis', n:int=3):
	"""
	This function shows an
		:param image_path: String or Path containing the images
		:param img_df: A dataframe listing all images used for training/testing
		:param get_labels: A function mapping from images to labels
		:param get_weights: A function mapping from images to weights
		:param n: Number of example batches to show
		:return: None
	
	"""
	
	assert n>0, 'n must be greater than 0.'
	assert n<=len(img_df), 'n must not exceed number of items.'
	
	if isinstance(cmap, str):
		cmap = 3*[cmap]
	
	else:
		assert len(cmap) == 3, 'cmap must be a list of length 3 or a single string'
	
	rndImages = img_df.sample(n)
	rndImages = rndImages.to_numpy()
	
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
	
		for img in rndImages:
			img = img[0]

			imgEx = io.imread(f'{image_path}/{img}')
			imgEx = pil2tensor(imgEx,np.float32)[0]

			maskEx = io.imread(get_labels(img))
			maskEx = pil2tensor(maskEx,np.float32)[0]

			weightEx = io.imread(get_weights(img))
			weightEx = pil2tensor(weightEx,np.float32)[0]

			f, axarr = plt.subplots(1,3,figsize=(20,10))
			plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
			axarr[0].imshow(imgEx,cmap=cmap[0])
			axarr[0].title.set_text(f'Image ({img})')
			axarr[1].imshow(maskEx,cmap=cmap[1])
			axarr[1].title.set_text('Labels')
			axarr[2].imshow(weightEx, cmap=cmap[2])
			axarr[2].title.set_text('Weights')


def build_grid(shape:tuple=(540,540),
			   main_divs:int=6,
			   fine_divs:int=5,
			   fine_div_decay:float=0.3):
	"""
	This function builds a grid layout to visualize augmentations.
		:param shape: The size/shape of the grid
		:param main_divs: The number of main divisions in the grid
		:param fine_divs: The number of fine divisions within one main division
		:param main_divs_decay: The visibility of the fine divisions
		:return: returns a numpy array
	
	"""
	total_divs = main_divs*fine_divs
	intervals = np.divide(shape, total_divs)

	# get divisions along x-axis
	xdivisions = []
	x =intervals[0]
	while x < shape[0]:
		xdivisions.append(int(x))
		x += intervals[0]

	# get divisions along y-axis
	ydivisions = []
	y =intervals[0]
	while y < shape[0]:
		ydivisions.append(int(y))
		y += intervals[0]

	# create numpy array
	origin = np.zeros(shape)
	for i,x in enumerate(xdivisions):
		for y in range(shape[1]):
			if (i+1)%fine_divs ==0:
				origin[x][y] = 1
			else:
				origin[x][y] = 0.3

	for i,y in enumerate(ydivisions):
		for x in range(shape[0]):
			if (i+1)%fine_divs ==0:
				origin[x][y] = 1
			else:
				origin[x][y] = 0.3

	return origin
	

def test_deformation(shape:tuple=(540,540), 
					 figsize:tuple=(14,7),
					 example_img=None,
					 sigma:int=8,
					 points:int=5):
	"""
	This function visualizes the magnitude of the _elastic_transform augmentation.
		:param shape: The size/shape of the image/augmentation
		:param example_img: An example image in form of a numpy array 
		:param sigma: The magnitude of deformation
		:param points: The number of grid points on the deformation grid
		:return: None
	
	"""
	# build grid if no example image is passed
	if example_img is None:

		interpolation_magnitude = 1
		origin = build_grid(shape)
	
	# otherwise use given image and crop to shape
	else:
		interpolation_magnitude = 0
		origin = example_img
		
		# if image is RGB
		if len(origin.shape)==3:
			starting_points = np.subtract(origin[0].shape, shape)
			
		# if image is B&W
		else:
			starting_points = np.subtract(origin.shape, shape)
			origin = np.expand_dims(origin,0)
		
		start_x = np.random.randint(0,starting_points[0])
		start_y = np.random.randint(0,starting_points[1])
		
		origin = origin[:, start_x:start_x+shape[0], start_y:start_y+shape[1]]
		

	#expand array dimensions
	if len(origin.shape) == 2:
		origin = np.expand_dims(origin,0)
		

	# do the elastic transform
	result = _elastic_transform(origin,
							    seed=42,
								sigma=sigma,
							    points=points,
							    interpolation_magnitude=interpolation_magnitude)
	
	# plot original image and resulting image
	f, axs = plt.subplots(1,2,figsize=figsize)
	axs[0].imshow(origin[0], cmap='binary', interpolation= 'bicubic')
	
	if example_img is None:
		axs[1].imshow(result[0], cmap='binary', interpolation= 'bicubic')
	else:
		axs[1].imshow(origin[0], cmap='binary', interpolation= 'bicubic')
		axs[1].imshow(result[0], cmap='pink_r', alpha=0.8,interpolation= 'bicubic')
	
		
"""User Input Functions"""

def chose_lr():
	lr = float(input('Chose learning rate (e.g. 1e-04): \n'))
	return lr
	
def chose_n_epochs():
	n_epochs = int(input('Number of training epochs: \n'))
	return n_epochs
	
def chose_fold():
	fold = int(input('Index of fold to use for further training (0,...,n): \n'))
	return fold



"""Functions to disable and enable printing - StackOverflow # 8391411 """

# Disable printing
def blockPrint():
	sys.stdout = open(os.devnull, 'w')

# Restore printing
def enablePrint():
	sys.stdout = sys.__stdout__

from deepflash.fastai_extension import _elastic_transform, _do_crop_y


