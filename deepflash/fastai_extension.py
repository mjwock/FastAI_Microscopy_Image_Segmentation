from fastai.vision import *
from fastai.vision.transform import _crop_image_points, _crop_default

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from skimage import io

import numpy as np

import elasticdeform	#!pip install elasticdeform
import pdb


# Helper functions for _elastic_transform from https://pypi.org/project/elasticdeform/
def _normalize_inputs(X):
		if isinstance(X, np.ndarray):
				Xs = [X]
		elif isinstance(X, list):
				Xs = X
		else:
				raise Exception('X should be a numpy.ndarray or a list of numpy.ndarrays.')

		# check X inputs
		assert len(Xs) > 0, 'You must provide at least one image.'
		assert all(isinstance(x, np.ndarray) for x in Xs), 'All elements of X should be numpy.ndarrays.'
		return Xs

def _normalize_axis_list(axis, Xs):
		if axis is None:
				axis = [tuple(range(x.ndim)) for x in Xs]
		elif isinstance(axis, int):
				axis = (axis,)
		if isinstance(axis, tuple):
				axis = [axis] * len(Xs)
		assert len(axis) == len(Xs), 'Number of axis tuples should match number of inputs.'
		input_shapes = []
		for x, ax in zip(Xs, axis):
				assert isinstance(ax, tuple), 'axis should be given as a tuple'
				assert all(isinstance(a, int) for a in ax), 'axis must contain ints'
				assert len(ax) == len(axis[0]), 'All axis tuples should have the same length.'
				assert ax == tuple(set(ax)), 'axis must be sorted and unique'
				assert all(0 <= a < x.ndim for a in ax), 'invalid axis for input'
				input_shapes.append(tuple(x.shape[d] for d in ax))
		assert len(set(input_shapes)) == 1, 'All inputs should have the same shape.'
		deform_shape = input_shapes[0]
		return axis, deform_shape


"""AUGMENTATIONS"""

class TfmCropY(TfmPixel):
	"Decorator for just cropping y's."
	order=100

def _do_crop_y(x,mask_size=(356,356)):
	rows,cols = tis2hw(mask_size)
	row = int((x.size(1)-rows+1) * 0.5)
	col = int((x.size(2)-cols+1) * 0.5)
	x = x[:, row:row+rows, col:col+cols].contiguous()
	return x

#wrapper for _do_crop_y
do_crop_y = TfmCropY(_do_crop_y)

def _initial_crop_pad(x, height=540, width=540, row_pct:uniform=0.5, col_pct:uniform=0.5):
	f_crop = _crop_image_points if isinstance(x, ImagePoints) else _crop_default
	return f_crop(x, (height,width), row_pct, col_pct)

initial_crop_pad = TfmPixel(_initial_crop_pad, order = 0)

def _elastic_transform(x, seed:uniform_int=42,sigma=8,points=5,interpolation_magnitude=0,mode="constant"):
	"""Elastic deformation based on following repo: 
			https://pypi.org/project/elasticdeform/
	"""
	image_array = np.asarray(x.data)[0]
	cval = 0.0
	prefilter=True
	axis=None
	if mode =="constant":
		cval=np.unique(image_array)[0]

	# prepare inputs and axis selection
	Xs = _normalize_inputs(image_array)
	axis, deform_shape = _normalize_axis_list(axis, Xs)

	if not isinstance(points, (list, tuple)):
		points = [points] * len(deform_shape)

	np.random.seed(seed=seed)
	displacement = np.random.randn(len(deform_shape), *points) * sigma

	image_array = elasticdeform.deform_grid(image_array, displacement=displacement, order=interpolation_magnitude, mode=mode,cval=cval, prefilter=prefilter, axis=axis)
	
	return pil2tensor(image_array,np.float32)

#wrapper for _elastic_transform
elastic_transform = TfmPixel(_elastic_transform)


#set random position range for random_crop and tile_shape
rand_pos_set = {'row_pct':(0.3,0.7), 'col_pct':(0.3,0.7)}
tile_shape_set = (540,540)

def get_custom_transforms(do_flip:bool=True,
						flip_vert:bool=True,
						elastic_deformation=True,
						elastic_deform_args={'sigma':10, 'points':10},
						random_crop= tile_shape_set,
						rand_pos = rand_pos_set,
						max_rotate:float=10.,
						max_lighting:float=0.2, 
						p_affine:float=0.75,
						p_lighting:float=0.75, 
						p_deformation = 0.7,
						transform_valid_ds =False,
						xtra_tfms:Optional[Collection[Transform]]=None) -> Collection[Transform]:
							
	"Utility func to easily create a list of flip, rotate, elastic deformation, lighting transforms."
	res = []
	if random_crop:res.append(initial_crop_pad(height=random_crop[0],width=random_crop[1],**rand_pos))
	if do_flip:    res.append(dihedral_affine() if flip_vert else flip_lr(p=0.5))
	if max_rotate: res.append(rotate(degrees=(-max_rotate,max_rotate), p=p_affine))
	if elastic_deformation: res.append(elastic_transform(p=p_deformation,seed=(0,1000),**elastic_deform_args))
	if max_lighting:
		res.append(brightness(change=(0.5, 0.5*(1+max_lighting)), p=p_lighting, use_on_y=False))
		res.append(contrast(scale=(1, 1/(1-max_lighting)), p=p_lighting, use_on_y=False))

	if transform_valid_ds: res_y = res 
	else: res_y=[initial_crop_pad(height=random_crop[0],width=random_crop[1],**rand_pos)]

	#       train                   , valid
	return (res + listify(xtra_tfms), res_y)


"""CUSTOM ItemBase"""

class WeightedLabels(ItemBase):
	"""Custom ItemBase to store and process labels and pixelweights together.
	Also handling the target_size of the masks.
	"""
	def __init__(self, lbl:Image, wgt:Image, target_size:Tuple=None):
		self.lbl,self.wgt = lbl,wgt
		self.obj,self.data = (lbl,wgt),[lbl.data, wgt.data]

		self.target_size = target_size

	def apply_tfms(self, tfms, **kwargs): 
		# if mask should be cropped, add operation 'do_crop_y' to transforms
		crop_to_target_size = self.target_size
		if crop_to_target_size:
			if not tfms:
				tfms = []
			tfms.append(do_crop_y(mask_size=crop_to_target_size))

		# transform labels and weights seperately
		self.lbl = self.lbl.apply_tfms(tfms, mode = 'nearest', **kwargs)
		self.wgt = self.wgt.apply_tfms(tfms, mode = 'nearest', **kwargs)
		self.data = [self.lbl.data, self.wgt.data]
		return self

	def __repr__(self):
		return f'{self.__class__.__name__}{(self.lbl, self.wgt)}'

"""CUSTOM ItemList AND ItemLabelList"""

class CustomSegmentationLabelList(ImageList):
	"'Item List' suitable for DeepFLaSH Masks"
	_processor= vision.data.SegmentationProcessor
	def __init__(self, 
							 items:Iterator, 
							 wghts=None,
							 classes:Collection=None,
							 target_size:Tuple=None,
							 loss_func=CrossEntropyFlat(axis=1),
							 **kwargs):
		
		super().__init__(items,**kwargs)
		self.copy_new.append('classes')
		self.copy_new.append('wghts')
		self.classes,self.loss_func,self.wghts = classes,loss_func, wghts
		self.target_size = target_size

	def open(self, fn): 
		res = io.imread(fn)
		res = pil2tensor(res,np.float32)
		return Image(res)

	def get(self, i):
		fn = super().get(i)
		wt = self.wghts[i]
		return WeightedLabels(fn,self.open(wt),self.target_size)
	
	def reconstruct(self, t:Tensor):
		return WeightedLabels(Image(t[0]),Image(t[1]),self.target_size)

	#def analyze_pred(self, pred, thresh:float=0.5): return pred.argmax(dim=0)[None]

class CustomSegmentationItemList(ImageList):
	"'ItemList' suitable for segmentation with pixelwise weighted loss"
	_label_cls,_square_show_res = CustomSegmentationLabelList,False

	def label_from_funcs(self, get_labels:Callable, get_weights:Callable,
											 label_cls:Callable=None, classes=None,
											 target_size:Tuple=None, **kwargs)->'LabelList':
		"Get weights and labels from two functions. Saves them in a CustomSegmentationLabelList"
		kwargs = {}
		wghts = [get_weights(o) for o in self.items]
		labels = [get_labels(o) for o in self.items]

		if target_size:
			print(f'Masks will be cropped to {target_size}. Choose \'None\' to keep initial size.')
		else:
			print(f'Masks will not be cropped.')
		
		y = CustomSegmentationLabelList(labels,wghts,classes,target_size,path=self.path)
		res = self._label_list(x=self, y=y)
		return res



	def show_xys(self, xs, ys, figsize:Tuple[int,int]=None, padding=184, **kwargs):
		"Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."

		if not figsize:
			figsize = (18,6*len(xs))

		#colormap for labels
		lmap = plt.get_cmap('Dark2')
		lmap.set_under('k', alpha=0)

		#colormap for weights
		wmap = plt.get_cmap('YlOrRd') #YlOrRd PuRd
		wmap.set_under('k', alpha=0)
		wmax = np.unique(ys[0].wgt.data)[-1]*0.9

		same_size = (ys[0].lbl.size == xs[0].size)

		#get width from x and set red boundingbox
		wdt, hgt = xs[0].size

		if not same_size:
			padding = int(0.5*(wdt - ys[0].lbl.size[1]))  #####numpy pad
		else:
			padding = int(0.5*padding)

														#(x-offset,y-offset)      ,width      ,height
		bbox = patches.Rectangle((padding-1,padding-1),wdt-2*padding,hgt-2*padding,edgecolor='r',linewidth=1,facecolor='none')

		#rows: number of batch items
		rows = len(xs)
		fig, axs = plt.subplots(rows,3,figsize=figsize)
		#idx: batch item counter
		idx = 0
		for i, ax in enumerate(axs.flatten() if rows > 1 else axs):
			#display image and labels together
			if(i%3==0):
				# normalize the images for a better visualization
				imax = np.unique(xs[idx].data)[-1]

				y_label = np.asarray(ys[idx].lbl.data[0])

				if not same_size:
					y_label = np.pad(y_label, padding, 'constant', constant_values=(0))

				ax.imshow(np.asarray(xs[idx].data[0]),vmax=imax, cmap='binary', interpolation='none') #cmap= bone_r
				ax.imshow(y_label, cmap=lmap, vmin=0.2, alpha=0.8, interpolation='none')
				ax.add_patch(copy(bbox))
				ax.set_title('Image + Labels')

			#display image and weights together
			elif(i%3==1):

				y_weights = np.asarray(ys[idx].wgt.data[0])

				if not same_size:
					y_weights = np.pad(y_weights, padding, 'constant', constant_values=(0))

				ax.imshow(np.asarray(xs[idx].data[0]), vmax=imax, cmap='binary', interpolation='none')
				ax.imshow(y_weights, vmax=wmax, cmap=wmap, vmin=0.15, alpha=0.5,interpolation='none')

				ax.add_patch(copy(bbox))
				ax.set_title('Image + Weights')

			#display labels and weights together       
			else:
				ax.imshow(y_weights,vmax=wmax, cmap=wmap,vmin=0.15,interpolation='none')
				ax.imshow(y_label, cmap=lmap,vmin=0.1,alpha=1,interpolation='none')
				ax.add_patch(copy(bbox))
				ax.set_title('Labels + Weights')

				idx +=1
		plt.tight_layout()

	def show_xyzs(self, xs, ys, figsize:Tuple[int,int]=None, padding=184, **kwargs):
		"Show the `xs`, `ys`and `zs` on a figure of `figsize`. `kwargs` are passed to the show method."

		if not figsize:
			figsize = (18,6*len(xs))

		#colormap for labels
		lmap = plt.get_cmap('Blues')
		lmap.set_under('k', alpha=0)

		pmap = plt.get_cmap('Reds')
		pmap.set_under('k', alpha=0)

		same_size = (ys[0].lbl.size == xs[0].size)

		#get width from x and set red boundingbox
		wdt, hgt = xs[0].size

		if not same_size:
			padding = int(0.5*(wdt - ys[0].lbl.size[1]))  #####numpy pad
		else:
			padding = int(0.5*padding)

														#(x-offset,y-offset)      ,width      ,height
		bbox = patches.Rectangle((padding-1,padding-1),wdt-2*padding,hgt-2*padding,edgecolor='r',linewidth=1,facecolor='none')

		#rows: number of batch items
		rows = len(xs)
		fig, axs = plt.subplots(rows,3,figsize=figsize)
		#idx: batch item counter
		idx = 0
		for i, ax in enumerate(axs.flatten() if rows > 1 else axs):
			
			#display image and labels together
			if(i%3==0):
				# normalize the images for a better visualization
				imax = np.unique(xs[idx].data)[-1]

				y_label = np.asarray(ys[idx].lbl.data[0])

				if not same_size:
					y_label = np.pad(y_label, padding, 'constant', constant_values=(0))

				ax.imshow(np.asarray(xs[idx].data[0]),vmax=imax, cmap='binary', interpolation='none') #cmap= bone_r
				ax.imshow(y_label, cmap=lmap, vmin=0.2, alpha=0.8, interpolation='none')
				ax.add_patch(copy(bbox))
				ax.set_title('Image + Labels')

			#display image and predictions together
			elif(i%3==1):

				y_predictions = np.asarray(zs[idx].data[0])

				if not same_size:
					y_predictions = np.pad(y_predictions, padding, 'constant', constant_values=(0))

				ax.imshow(np.asarray(xs[idx].data[0]), vmax=imax, cmap='binary', interpolation='none')
				ax.imshow(y_predictions, vmax=wmax, cmap=wmap, vmin=0.1, alpha=0.8,interpolation='none')

				ax.add_patch(copy(bbox))
				ax.set_title('Image + Predictions')

			#display labels and weights together       
			else:
				ax.imshow(y_predictions,vmax=wmax, cmap=wmap,vmin=0.1,alpha=0.6,interpolation='none')
				ax.imshow(y_label, cmap=lmap,vmin=0.1,alpha=0.6,interpolation='none')
				ax.add_patch(copy(bbox))
				ax.set_title('Labels + Predictions')

				idx +=1
		plt.tight_layout()
		
		
"""CUSTOM LOSS"""

class FlattenedWeightedLoss():
		"Same as `func`, but flattens input and target."
		def __init__(self, func, *args, 
								 axis:int=1,
								 reduction_mode='sum', 
								 longify:bool=True, 
								 is_2d:bool=True, 
								 **kwargs):
				self.func,self.axis,self.longify,self.is_2d = func(*args,**kwargs),axis,longify,is_2d
				self.reduction_mode = reduction_mode

		def __repr__(self): return f"My FlattenedLoss of {self.func}"
		@property
		def reduction(self): return self.func.reduction
		@reduction.setter
		def reduction(self, v): self.func.reduction = v

		def __call__(self,
					 input:Tensor,
					 labels:Tensor, 
					 weights:Tensor,
					 **kwargs) -> Rank0Tensor:
				
				assert self.reduction_mode in ('sum','mean','none'), \
					'Check reduction_mode and chose between sum, mean and none'
				
				#pdb.set_trace()
												
				# flatten
				input = input.transpose(self.axis,-1).contiguous()
				labels = labels.transpose(self.axis,-1).contiguous()
				weights = weights.transpose(self.axis,-1).contiguous()

				# transform to long
				if self.longify: 
					labels = labels.long()       

				# reshape
				input = input.view(-1,input.shape[-1]) if self.is_2d else input.view(-1)
				labels = labels.view(-1)
				weights = weights.view(-1)
				
				#pdb.set_trace()

				res = nn.CrossEntropyLoss(reduction='none')(input, labels, **kwargs)

				if self.reduction_mode =='sum':
					return (weights * res).sum()
				elif self.reduction_mode =='mean':
					return (weights * res).mean()
				else:
					return weights * res

def WeightedCrossEntropyLoss(*args, axis:int=-1, **kwargs):
		"Same as `nn.CrossEntropyLoss`, but flattens input and target."
		return FlattenedWeightedLoss(nn.CrossEntropyLoss, *args, axis=axis, **kwargs)
		

"""METRICS"""

class flattened_metrics():
	"""Class to handle regular loss functions, pass any given kwargs and ignore the passed weights"""
	def __init__(self,func:Callable, swap_pred=False, softmax=True, argmax=False, **kwargs):
		self.func = func
		self.kwargs = kwargs
		self.swap_pred = swap_pred
		self.softmax = softmax
		self.argmax = argmax
	
	def __repr__(self): return f"Wrapper for {self.func}"

	def __call__(self,input:Tensor, target:Tensor,weights:Tensor):
		
		if self.softmax:
			input = nn.Softmax2d()(input)
		
		if self.argmax:
			input = nn.argmax(input)
				
		if self.swap_pred:
			res = self.func(target,input,**self.kwargs)
		else:
			res = self.func(input,target,**self.kwargs)

		return res

def metrics_wrapper(*args, metric:Callable, swap_pred=False, softmax=True,argmax=False, **kwargs):
	"""
	Wrapper for any metric that takes predictions and labels to evaluate training.	
	Args:
		:metric: desired loss function
		:name: name that is displayed in fastai
		:swap_pred: bool value to swap prediction with ground_truth (e.g. for sklearn losses)
		:softmax:
		any additional kwargs will be passed into the loss function
	"""
	return flattened_metrics(*args, func=metric, swap_pred=swap_pred, softmax=softmax,argmax=argmax, **kwargs)
