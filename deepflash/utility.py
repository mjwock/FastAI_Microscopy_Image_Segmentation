from fastai.vision import *
from skimage import io
from tqdm.auto import tqdm

from deepflash.fastai_extension import _elastic_transform, _do_crop_y

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import pandas as pd
import seaborn as sns

import warnings
import sys, os, re

import shutil

def purge(dir, pattern):
	"""Delete all files following a pattern"""
	for f in os.listdir(dir):
		if re.search(pattern, f):
			os.remove(os.path.join(dir, f))

def split_net(model):
	"""Split model on the next hierarchical layer and returns them as lists."""
	return list(model.children())
	
def load_images_and_labels(image_dir,label_dir):
	"""Load images and labels from directory"""
	image_list = os.listdir(image_dir)
	label_list = os.listdir(label_dir)
	
	image_list.sort()
	label_list.sort()
	
	names = [os.path.splitext(image)[0] for image in image_list]
	
	images = [Image(pil2tensor(io.imread(f'{image_dir}/{img}', as_gray=True), np.float32)) for img in image_list]
	labels = [Image(pil2tensor(io.imread(f'{label_dir}/{lbl}', as_gray=True), int)) for lbl in label_list]

	return images, labels, names
	
def get_image_size(path):
	"""
	Returns image size of image at 'path' location
	:param path: path to target image
	"""
	imgEx = io.imread(path,as_gray=True)
	return imgEx.shape

def show_example_data_batch(image_path, img_df:DataFrame, get_labels:Callable, get_weights:Callable,cmap='viridis', n:int=3):
	"""
	This function shows an example batch consisting of image, label and weights
		:param image_path:	String or Path containing the images
		:param img_df: 		A dataframe listing all images used for training/testing
		:param get_labels: 	A function mapping from images to labels
		:param get_weights: A function mapping from images to weights
		:param n: 			Number of example batches to show
		:return: 			None
	
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
		:param shape: 			The size/shape of the grid
		:param main_divs: 		The number of main divisions in the grid
		:param fine_divs: 		The number of fine divisions within one main division
		:param main_divs_decay: The visibility of the fine divisions
		:return: 	 			A numpy array in grid shape
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
					 points:int=5,
					 seed:int=42):
	"""
	This function visualizes the magnitude of the _elastic_transform augmentation.
		:param shape: 		The size/shape of the image/augmentation
		:param example_img: An example image in form of a numpy array 
		:param sigma: 		The magnitude of deformation
		:param points: 		The number of grid points on the deformation grid
		:param seed: 		Seed for np.random	
		:return: 			None
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
							    seed=seed,
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

def chose_lr(discriminative_lr:bool=False):
	"""
	:param discriminative_lr: 	If True, lr is automatically transferred to discriminative lr slice
	:return: 					Slice of user input learning rate
	"""
	lr_string = input('Chose learning rate(s) (e.g. \'1e-04\' or \'1e-04,1e-03\'): \n')
	
	list_of_lr = lr_string.split(',')
	
	if discriminative_lr and len(list_of_lr) == 1:
	# in case of discriminative lr and a single given lr
	
		lr = slice(0.1*float(list_of_lr[0]),float(list_of_lr[0]))
		# create a slice from max_lr/10 to max_lr 
		
	else:
	# otherwise
	
		lr = slice(*list(map(float, list_of_lr)))
		# map all strings in 'list_of_lr' to float
	
	return lr
	
def chose_n_epochs():
	""":return: int of user input number of epochs"""
	n_epochs = int(input('Number of training epochs: \n'))
	return n_epochs
	
def chose_fold():
	""":return: int of user input fold index"""
	fold = int(input('Please chose index of best fold for further training (0,...,n): \n'))
	return fold
	
def save_cycle(model_dir:str, cycle:int, fold:bool):
	"""
	Deletes the respective folder if cycle is discarded. 
	:param model_dir: 	Directory of the model
	:param cycle: 		Last cycle count
	:param fold: 		K-fold cross validation
	:Returns: 			New cycle count (int)
	"""
	decision = input('Should this run be saved? (y/n): \n')
	while not (decision == 'y' or decision == 'n'):
		decision = input('Please enter \'y\' or \'n\': \n')
	
	if decision == 'y':
	# if run should be kept
		if fold:
		# in case of k-fold
			fold_n = chose_fold()
			# chose best fold
			
			if os.path.exists(f'{model_dir}/{cycle}/metrics_{fold_n}.csv'):
			# delete all but target fold	
				os.rename(f'{model_dir}/{cycle}/temp_model_{fold_n}.pth',f'{model_dir}/{cycle}/temp_model.pth')
				purge(f'{model_dir}/{cycle}',r'temp_model_\d+.pth')
				print(f'Saved fold number {fold_n}.')	
			else:
				print(f'Could not find the file for fold number {fold_n}. To continue training, change the file manually to \'temp_model.pth\' \
				\n (Missing file: \'{model_dir}/{cycle}/metrics_{fold_n}.csv\'')
			
			if os.path.exists(f'{model_dir}/{cycle}/metrics_{fold_n}.csv'):
			# delete all but target metrics
				os.rename(f'{model_dir}/{cycle}/metrics_{fold_n}.csv',f'{model_dir}/{cycle}/metrics.csv')
			else:
				print(f'Warning! Cannot find \'metrics_{fold_n}.csv.\'')
			
		cycle +=1
		#increase the cycle count			
	else:
	# else discard run
		shutil.rmtree(f'{model_dir}/{cycle}')
		print('Run discarded.')
	
	return cycle		
	

"""Functions to disable and enable printing - StackOverflow # 8391411 """

# Disable printing
def blockPrint():
	sys.stdout = open(os.devnull, 'w')

# Enable printing
def enablePrint():
	sys.stdout = sys.__stdout__
	
	
"""Functions for Data Preparator"""

def calc_vbal(mask_path:str):
	"""
	Function to compute the weighting factor vbal depending on the class imbalance.
		:param masks: 	List of arrays containing the masks
		:return: 		Float value of v_bal 
	"""
	
	filenames = os.listdir(mask_path)
	mask_list = [io.imread(f'{mask_path}/{mask}', as_gray=True).astype('int') for mask in filenames]
	
	counter = None
	
	#for each mask count unique values and add to counter
	for mask in mask_list:
		unique, counts = np.unique(mask, return_counts=True)

		if counter is not None:
			counter = np.add(counter,counts)
		else:
			counter = counts  
	
	assert len(counter) > 1, 'Only background pixels in masks!'
	
	if len(counter == 2):
		v_bal = counter[1]/counter[0]
	else:
		v_bal = sum(counter[1:])/counter[0]
	
	print(f'Foreground to background ratio is {v_bal}')
	return v_bal


"""EVALUATION"""

class object_based_accuracy():
	'''
	Class to manage object based accuracy calculation as descibed by Caicedo, Roth, et al. (2019):
	Finds all secluded pixel areas from the prediction and labels and compares their intersect over union pairwise.
	False positive, false negative and true negative predictions are then used to compute a final F-Beta score.
	(https://onlinelibrary.wiley.com/doi/pdf/10.1002/cyto.a.23863)
		:param preds:     	list of predictions (2D arrays or PIL Image files)
		:param targs:     	list of targets (2D arrays or PIL Image files)
		:param iou_thresh:  threshold for the min overlap to count as same object
		:param beta: 		beta for the weighted F-measure
		:param size_thresh:	ignore all cell areas smaller than size_thresh 
		:param pred_cells:	add saved list with predicted cells (save list by retrieving self.pred_cells) 
		:param targ_cells:	add saved list with target cells	(save list by retrieving self.targ_cells) 
	'''
	def __init__(self, preds, targs, iou_thresh:float=0.5, beta:float = 1, size_thresh:int = None, pred_cells=None, targ_cells=None):
		self.preds = [pred.data[0] for pred in preds] if (type(preds[0]) == Image) else preds
		self.targs = [targ.data[0] for targ in targs] if (type(targs[0]) == Image) else targs
		
		self.size_thresh = size_thresh
		self.beta = beta
		self.iou_thresh = iou_thresh

		if pred_cells:
			self.pred_cells, self.targ_cells = pred_cells,targ_cells
		else:
			self.pred_cells, self.targ_cells = self.flood_cells()

		# compute number of identified cells and number of true cells
		self.n_preds = [len(pred) for pred in self.pred_cells] if size_thresh else [len(pred) for pred in self.pred_cells]
		self.n_targs = [len(targ) for targ in self.targ_cells] if size_thresh else [len(targ) for targ in self.targ_cells]

		# build iou matrix for all instances
		self.iouMatrices = [self.compare_cells(p,t) for p,t in tqdm(zip(self.pred_cells,self.targ_cells),desc='Building IoU Matrices.')]

		# compute confusion values for all predictions and non-predictions
		self.confusion_values = [self.compare_targs_preds(i, ioum,iou_thresh,size_thresh) for i, ioum in enumerate(self.iouMatrices)]

		# compute accuracies for each instance based on confusion values
		self.accuray = np.array([self.accuracys(conf, beta) for conf in self.confusion_values]).transpose()

		# generate confusion maps to 
		self.confusion_maps = [self.build_confusion_map(i) for i in range(len(self.preds))]
		
	def __call__(self):
		'''
		Returns mean, std, variance for precision, recall, fscore
		'''
		metrics = []
		
		for met in self.accuray:
			metrics.append([met.mean(),met.std(),met.var()])	
			
		#metrics = np.array(metrics)
		metrics_list = ['precision','recall',f'F{self.beta}']
			
		return pd.DataFrame(metrics,index=metrics_list,columns=['mean','std','var'])
			
	def save_to_csv(self,directory:str):
		'''
		Saves accuracys to csv
			:param directory: target directory for saving csv file
		'''	
		
		df = self.pandas_df()
		df.to_csv(f'{directory}/object_metrics_thresh({self.iou_thresh}).csv')
		
	def boxplot(self,figsize=(12,8)):
		'''
		Draw boxplot for all metrics
			:param figsize:	total size of the graph
		'''

		df = self.pandas_df()

		plt.figure(figsize=figsize, dpi= 80)
		sns.boxplot(x='metric',y='score',data = df, notch=False)   
		sns.stripplot(x='metric', y='score', data=df, color='black', size=3, jitter=1,alpha=0.5)

		for i in range(len(df['metric'].unique())-1):
			plt.vlines(i+.5, 0.1, 0.9, linestyles='solid', colors='gray', alpha=0.2)

		# Decoration
		plt.title('Object Based Metrics', fontsize=22)
		plt.ylim(0, 1)
		plt.show()
		
	
	def pandas_df(self):
		'''
		Return a pandas df in tidy long format
			:return:	DataFrame
		'''
		
		df_format = []
		for row,metric_name in zip(self.accuray,['precision','recall',f'F{self.beta}']):
			for entry in row:
				df_format.append([entry,metric_name])
		
		return pd.DataFrame(df_format,columns=['score','metric'])


	def flood_cells(self):
		'''
		Use the floodfill algorithm on all predictions and labels to detect
		how many cells are present in the images.
		'''

		pred_cells = []
		targ_cells = []

		for pred in tqdm(self.preds,desc='Flooding predictions.'):
			pred_cells.append(self.floodfill(pred))
		
		for targ in tqdm(self.targs,desc='Flooding targets.'):
			targ_cells.append(self.floodfill(targ))

		return pred_cells, targ_cells
		
	
	def floodfill(self,label):
		'''
		Simple floodfill algorithm that identifies enclosed areas
			:param label: a two dimensional array containing class values
			:return:      a list of cells with pixel coordinates
		'''

		seen = set()
		cells = []
		for i in range(len(label)):
				for j in range(len(label[i])):
					if label[i][j]>0:
						if (i,j) not in seen:
								# start flood-fill from that position
								same = [(i,j)]
								pos = 0
								while pos < len(same):
										x, y = same[pos]
										for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
												if 0 <= x2 < len(label) and  0 <= y2 < len(label[x2]) and \
																(x2,y2) not in seen and label[x2][y2] == label[i][j]:
														same.append((x2,y2))
														seen.add((x2,y2))
										pos += 1
								cells.append(same)
			
		return cells
	
	
	def compare_targs_preds(self,index, iou_matrix,iou_thresh,size_thresh=None):
		'''
		Compares the values in IoU matrix with a given threshold
			:param iou_matrix:  matrix containing all IoU combinations
			:param iou_thresh:  threshold for when an overlap counts as a match
			:return:            array with true positives, false negatives and false positives
		'''
		preds,targs = iou_matrix.shape

		TP = np.zeros(preds,np.int)
		FN = np.ones(targs,np.int)
		FP = np.ones(preds,np.int)

		for row in range(preds):
			for col in range(targs):
				if iou_matrix[row,col] > iou_thresh:
					TP[row] += 1
					FN[col],FP[row] = 0,0
					
		if size_thresh:
			for i, cell in enumerate(self.pred_cells[index]):
				if len(cell)<size_thresh:
					TP[i] = 0
					FP[i] = 0
						
		return [TP, FN, FP]

	
	def compare_cells(self,preds,targs):
		'''
		Compare the predictions to the ground truth by Intersect over Union (IoU).
			:param preds: array with predicted cells
			:param targs: array with true cells
			:param tresh: threshold to determine a match
			:return:      IoU matrix
		'''

		iouMatrix = np.zeros((len(preds),len(targs)))
		for i, pred in enumerate(preds):
			for j, targ in enumerate(targs):
				intersect = set(pred) & set(targ)
				union = set(pred+targ)

				iouMatrix[i,j] = len(intersect)/len(union)

		return iouMatrix

	def accuracys(self, confusion_values, beta):
		'''
		Calculate the f_beta score based on TP, FN, FP and beta for a single prediction.
			:param confusion_values:  array containing the confusion matrix values for f_beta
			:param beta:              weight for precision over recall (Dice for 'beta == 1')
			:return:                  [precision,recall,f_beta]
		'''

		TP, FN, FP = confusion_values
		if (TP.sum() + FP.sum()) == 0:
			precision = 0
		else:
			precision = TP.sum() / (TP.sum() + FP.sum()) 
			
		if (TP.sum() + FN.sum()) == 0:
			recall = 0
		else:
			recall = TP.sum() / (TP.sum() + FN.sum())
			
		if recall == 0 or precision == 0:
			f_score = 0.
		else:
			f_score = (1+beta)*(precision*recall)/(precision+recall)

		return [precision,recall,f_score]

	def reconfigure_matches(self,iou_thresh:float,beta:int=1,size_thresh=None):
		'''
		Recalculate accuracy measures based on new threshold.
			:param iou_thresh:  new threshold for matches
			:param beta:    new beta value for the f-score
		'''
		
		self.beta = beta
		self.iou_thresh = iou_thresh
		self.size_thresh = size_thresh 
		
		# compute confusion values for all predictions and non-predictions
		self.confusion_values = [self.compare_targs_preds(i, ioum,iou_thresh,size_thresh) for i, ioum in enumerate(self.iouMatrices)]

		# compute accuracies based on confusion values
		self.accuray = np.array([self.accuracys(conf, beta) for conf in self.confusion_values]).transpose()

		# generate confusion maps to 
		self.confusion_maps = [self.build_confusion_map(i) for i in range(len(self.preds))]

	def build_confusion_map(self, index):
		'''
			Creates a visual representation of TP, FP, FN and TN areas.
				:param index: index of subject
				:return:      returns an Image of a confusion map
		'''

		shape = self.preds[index].shape
		TP,FN,FP = self.confusion_values[index]
		map = np.zeros(shape,np.int)
		
		# add all false negative cells
		for i, fn in enumerate(FN):
			if fn:
				for x,y in self.targ_cells[index][i]:
					map[x][y] = 2

		# add all false positive cells
		for i, fp in enumerate(FP):
			if fp:
				for x,y in self.pred_cells[index][i]:
					map[x][y] = 1

		 # add all true positive cells
		for i, tp in enumerate(TP):
			if tp:
				for x,y in self.pred_cells[index][i]:
					map[x][y] = 3

		return Image(pil2tensor(map,dtype=int))
	
		
class evaluator():
	'''
	Class to calculate given metrics for a given list of predictions and labels.
		:param preds:     list of predictions (2D arrays or PIL Image files)
		:param targs:     list of targets (2D arrays or PIL Image files)
		:param metrics:   list of metrics (funcs or partials)
		:param conf_maps: calculate confusion maps if True (very slow!)
	'''

	def __init__(self, preds, targs, metrics,probabilities=None, conf_maps=False):

		self.preds = [pred.data[0] for pred in preds] if (type(preds[0]) == Image) else preds
		self.targs = [targ.data[0] for targ in targs] if (type(preds[0]) == Image) else targs

		self.metrics = metrics
		self.confusion_maps = None

		self.evaluations = [self.metric_calculation(self.preds,self.targs,metric) for metric in tqdm(self.metrics, 
																																																 desc='Calculating metrics...',
																																																 position=1)]
		if conf_maps:
			self.confusion_maps = [self.build_confusion_maps(pred, targ) for pred,targ in tqdm(zip(self.preds,self.targs), 
																																												desc='Building confusion maps...',
																																												total=len(self.preds))]

	def __call__(self):
		'''
		Returns DataFrame with aggregated metrics (mean,std,var) upon call
			:return:  DataFrame
		'''

		aggregations = []

		for eval in self.evaluations:
			aggregations.append([np.mean(eval), np.std(eval), np.var(eval)])

		aggregations = np.array(aggregations)
		metrics_list = [metric.func.__name__  if (type(metric) == functools.partial) else metric.__name__ for metric in self.metrics]

		return pd.DataFrame(aggregations,index=metrics_list,columns=['mean','std','var'])                                                                                   
	
	def metric_calculation(self, preds, targs, func):
		'''
		Calculates the accuracy for a list of predictions and targets based on func
			:param preds: list of predictions
			:param targs: list of labels
			:param func:  metric function

			:return:      list of measures
		'''

		accs = []

		for pred, targ in tqdm(zip(preds,targs), position=0,total=len(preds)):
			
			acc = func(torch.flatten(pred),torch.flatten(targ))
			accs.append(acc)

		return accs

	def save_to_csv(self, directory:str):
		'''
		Saves accuracys to csv.
			:param directory: target directory for saving csv file (str)
		'''	
		df = self.pandas_df()
		df.to_csv(f'{directory}/simple_metrics.csv')

	def boxplot(self,figsize=(12,8)):
		'''
		Draws a boxplot for all metrics.
			:param figsize: size of figure (tuple)
		'''
		
		df = self.pandas_df()

		plt.figure(figsize=figsize, dpi= 80)
		sns.boxplot(x='metric',y='score',data = df, notch=False)   
		sns.stripplot(x='metric', y='score', data=df, color='black', size=3, jitter=1,alpha=0.5)

		for i in range(len(df['metric'].unique())-1):
				plt.vlines(i+.5, 0.1, 0.9, linestyles='solid', colors='gray', alpha=0.2)

		# Decoration
		plt.title('Metrics', fontsize=22)
		plt.ylim(0, 1)
		plt.show()

	def pandas_df(self):
		'''
		Transforms data into long format to be used for plotting and saving as DataFrame
			:return:  DataFrame
		'''

		df_format = []
		for i, row in enumerate(np.array(self.evaluations)):
			metric = self.metrics[i]
			metric_name = metric.func.__name__  if (type(metric) == functools.partial) else metric.__name__ 
			for entry in row:
				df_format.append([entry,metric_name])

		return pd.DataFrame(df_format,columns=['score','metric'])

	def build_confusion_maps(self, preds, targs):
		'''
		Creates pixelwise confusion maps displaying TPs, FPs, FNs, TNs
			:param preds: prediction item
			:param targs: label item

			:return:      PIL Image of confusion map
		'''
		import pdb
		shape = preds.shape
		map = np.zeros(shape,np.int)
		#pdb.set_trace()
		for row in range(shape[0]):
			for col in range(shape[1]):
				if preds[row][col] == 0:
					# FN
					if targs[row][col] == 1:
						map[row][col] = 2
				else:
					# TP
					if targs[row][col] == 1:
						map[row][col] = 3
					# FP
					else:
						map[row][col] = 1
						
		
		return Image(pil2tensor(map,dtype=float))

	def get_confusion_map(self,index):
		'''Return confusion map of index (Image)'''
		if self.confusion_maps:
			return self.confusion_maps[index]
		else:
			return self.build_confusion_maps(self.preds[index],self.targs[index])
			
	def plot_conf_map(self, index, figsize= (12,12)):
		'''Plot confusion map of index in a colorblind and print friendly manner'''
		
		#define color space
		cm_data = ['#a6cee3','#1f78b4','#b2df8a','#33a02c']
		cmap = sns.color_palette(cm_data) 

		plt.figure(figsize=figsize, dpi= 80)
		
		#get confusion map
		conf_map = self.get_confusion_map(index).data[0]
		
		# plot heatmap	
		ax = sns.heatmap(conf_map, cmap=cmap, square=True, yticklabels=[], xticklabels=[], cbar_kws={"shrink": 0.82}) 
		
		# change colorbar
		value_to_int = {j:i for i,j in enumerate(['TN','FP','FN','TP'])}
		n = len(value_to_int) 
		col_bar = ax.collections[0].colorbar
		r = col_bar.vmax - col_bar.vmin 
		col_bar.set_ticks([col_bar.vmin + r / n * (0.5 + i) for i in range(n)])
		col_bar.set_ticklabels(list(value_to_int.keys()))  
		
		plt.show()
		
	def plot_prob_map(self, probabilities, figsize=(12,12)):
		'''
		Function to plot probability heatmap
			:param probabilities:	Image file of probabilities or 2D array
		'''
		if type(probabilities == Image):
			probabilities = probabilities.data[0]
		
		plt.figure(figsize=figsize, dpi= 80)
		xd,yd = probabilities.shape
		ax = sns.heatmap(probabilities,cmap=cm.GnBu,square=True,cbar_kws={"shrink": (yd/xd)*0.82},yticklabels=[],xticklabels=[]) 

class inter_coding_reliability():
	'''
	Manages the Inter Coding Reliability
	'''
	def __init__(self, preds, dir, sub_dirs, name_list):

		self.preds = [pred.data[0] for pred in preds] if (type(preds[0]) == Image) else preds
		self.dir = f'/{dir.strip("/")}'
		self.sub_dirs = sub_dirs
		self.name_list = name_list

		self.dim = len(self.sub_dirs)+1

		self.coder_labels = [self.get_files(f'{self.dir}/{coder}',name_list) for coder in tqdm(sub_dirs,desc='Loading Coders',position=1)]
		# scores in matrix mapping from 'sub_dirs ->  -sub_dirs'
		self.score_matrix = None

	def __call__(self, metric):
		'''
		Calculate metric upon call, return mean values as pandas df
			:param metric:  func of metric

			:return:        DataFrame
		'''

		score_matrix = []

		# calculate first row with predictions and all labels
		row = []
		for labels in self.coder_labels:
			row.append(self.calc_score(self.preds,labels,metric))

		score_matrix.append(row)

		# calculate all other label pairs
		for i, a in tqdm(enumerate(self.coder_labels), total=len(self.coder_labels)):

			# collect all row elements
			row = []
			for j, b in enumerate(self.coder_labels):
				#coder -> coder danach
				# only compare those that haven't been compared yet
				if i<j:
					row.append(self.calc_score(a,b,metric))

			# add to score_matrix
			score_matrix.append(row)
		
		# save results to score_matrix
		self.score_matrix = score_matrix

		#return df with mean values
		return self.to_df()
		
	def save_to_csv(self,model_dir:str):
		'''
		Saves results to 'inter_coding.csv' in model_dir
			:param model_dir:	directory to save the model in
		'''
		
		df = self.to_df()
		df.to_csv(f'{directory}/inter_coding.csv')
		
	def to_df(self):
		'''
		Create df from score matrix
			:return: DataFrame representation of the score_matrix
		'''
		
		# new matrix for mean values
		mean_matrix = np.zeros((self.dim,self.dim))
		labels = ['prediction'] + self.sub_dirs

		# reorganize values for presentation
		for i, coder in enumerate(self.score_matrix):
			for j, values in enumerate(coder):
				mean_matrix[i][-(j+2+i)] = np.mean(values)
		
		return pd.DataFrame(mean_matrix,index=labels, columns=labels[::-1])
		

	def plot_results(self, figsize:tuple = (12,12),anon:bool=False):
		'''
		Plot results in a matrix with accuracy measures top left and plots bottom right
			:param figsize:	size of plot 
			:param anon:	anonymize names of experts
		'''
		
		f, axarr = plt.subplots(self.dim,self.dim,figsize=figsize)
		plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
		plt.ylim(0, 1)

		# get axes labels
		if anon:
			labels = ['Prediction'] + [f'Expert{n}' for n,_ in enumerate(self.sub_dirs)]
		else:
			labels = ['prediction'] + self.sub_dirs
		

		# for all rows
		for i, coder in enumerate(self.score_matrix):
			# set ticks
			axarr[i,-1].yaxis.tick_right()
			axarr[i,-1].set_yticks([0,0.5,1])
			[l.set_visible(False) for l in axarr[i,0].get_yticklabels()]

			# get all columns
			for j, values in enumerate(coder):
				mean_val = np.mean(values)

				# index for restructuring
				j_index = -(j+2+i)

				# plot all textboxes with congruent colors
				axarr[i,j_index].text(0.5, 0.5, str(np.round(mean_val,2)), horizontalalignment='center', verticalalignment='center')
				axarr[i,j_index].set_facecolor(cm.GnBu_r(mean_val))

				# violin plot mirrored to textboxes
				sns.violinplot(values,scale='width', inner='quartile',orient='v', ax=axarr[-(j_index+1),-(i+1)])  

				# set labels
				if i == 0:
					axarr[i, j].set_title(labels[-j-1])
				if j == 0:
					axarr[i, j].set_ylabel(labels[i])
				
			# set last column label
			if i == 0:
				axarr[0, len(labels)-1].set_title(labels[0])

			# set last row label
			if i == (len(labels)-1):
				axarr[i, 0].set_ylabel(labels[len(labels)-1])

			axarr[i, -1].yaxis.set_ticks([0,0.5,1])
			axarr[i, -1].yaxis.tick_right()
			axarr[i, -1].yaxis.set_visible(True)


	def calc_score(self, preds, targs, func):
		'''
		Calculate scores for a list of predictions and targets based on func
			:param preds: list of predictions (tensors)
			:param targs: list of labels (tensors)
			:param func:  metric function

			:return:      list of accuracys
		'''
		
		accs = []
		
		# for each prediction/label pair
		for pred, targ in zip(preds,targs):
			
			# calculate accuracy and add to accs
			acc = func(torch.flatten(pred),torch.flatten(targ))
			accs.append(acc)
		
		return accs

	def load_label(self,file):
		'''
		Load file to 2D tensor
			:param file:  Path to file (str)

			:return:      2D tensor
		'''
		
		return pil2tensor(io.imread(file, as_gray=True), np.float32)[0]
	
	def get_files(self, dir, name_list):
		'''
			Get all files in name_list from a directory in the correct load order
				:param dir:       directory with files (str)
				:param name_list: list of file_names

				:return:          list of labels (2D tensors)
		'''
		
		files = os.listdir(dir)
		files.sort()

		# empty load list for file names to lload
		load_list = []

		# get filenames in correct load order
		for name in name_list:
			for file in files:
				if name in file:
					load_list.append(file)

		# empty labels to save images
		labels = []
		# load all files
		for file in tqdm(load_list,desc='Loading Masks',position=0):
			labels.append(self.load_label(f'{dir}/{file}'))

		return labels