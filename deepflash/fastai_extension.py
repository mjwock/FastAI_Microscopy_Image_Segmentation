from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.transform import _crop_image_points, _crop_default

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import io

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm.auto import tqdm
from math import ceil
from copy import deepcopy

import numpy as np

import elasticdeform  # !pip install elasticdeform

import shutil
import warnings


# Helper functions for _elastic_transform from https://pypi.org/project/elasticdeform/
def _normalize_inputs(X):
    if isinstance(X, np.ndarray):
        Xs = [X]
    elif isinstance(X, list):
        Xs = X
    else:
        raise Exception(
            'X should be a numpy.ndarray or a list of numpy.ndarrays.')

    # check X inputs
    assert len(Xs) > 0, 'You must provide at least one image.'
    assert all(isinstance(x, np.ndarray)
               for x in Xs), 'All elements of X should be numpy.ndarrays.'
    return Xs


def _normalize_axis_list(axis, Xs):
    if axis is None:
        axis = [tuple(range(x.ndim)) for x in Xs]
    elif isinstance(axis, int):
        axis = (axis,)
    if isinstance(axis, tuple):
        axis = [axis] * len(Xs)
    assert len(axis) == len(
        Xs), 'Number of axis tuples should match number of inputs.'
    input_shapes = []
    for x, ax in zip(Xs, axis):
        assert isinstance(ax, tuple), 'axis should be given as a tuple'
        assert all(isinstance(a, int) for a in ax), 'axis must contain ints'
        assert len(ax) == len(
            axis[0]), 'All axis tuples should have the same length.'
        assert ax == tuple(set(ax)), 'axis must be sorted and unique'
        assert all(0 <= a < x.ndim for a in ax), 'invalid axis for input'
        input_shapes.append(tuple(x.shape[d] for d in ax))
    assert len(set(input_shapes)) == 1, 'All inputs should have the same shape.'
    deform_shape = input_shapes[0]
    return axis, deform_shape


"""AUGMENTATIONS"""


class TfmCropY(TfmPixel):
    '''
    Decorator for cropping the masks in last order in the augmentation pipeline.
    '''
    order = 100


def _do_crop_y(x, mask_size: tuple = (356, 356)):
    '''
    Centercrop the masks to size 'mask_size'
        :mask_size: tuple
    '''
    rows, cols = tis2hw(mask_size)
    row = int((x.size(1)-rows+1) * 0.5)
    col = int((x.size(2)-cols+1) * 0.5)
    x = x[:, row:row+rows, col:col+cols].contiguous()
    return x


# wrapper for _do_crop_y
do_crop_y = TfmCropY(_do_crop_y)


def _initial_crop_pad(x, height: int = 540, width: int = 540, row_pct: uniform = 0.5, col_pct: uniform = 0.5):
    '''
    Initial crop function with slightly different input arguments to perform a random crop.
        :height: height of the cropped area in pixels (int)
        :width: width of the cropped area in pixels (int)
        :row_pct: percentage for horizontal center point or uniform range for random placement (float)
        :col_pct: percentage for vertical center point or uniform range for random placement (float)
    '''
    f_crop = _crop_image_points if isinstance(
        x, ImagePoints) else _crop_default
    return f_crop(x, (height, width), row_pct, col_pct)


# wrapper for _initial_crop_pad
initial_crop_pad = TfmPixel(_initial_crop_pad, order=0)


def _elastic_transform(x, seed: uniform_int = 42, sigma=8, points=5, interpolation_magnitude=1, mode="mirror"):
    '''
    Slightly altered elastic smooth deformation based on following repo: https://pypi.org/project/elasticdeform/
        :seed: random factor for np.random, uniform_int range for random seeding (int)
        :sigma: standard deviation of the normal distribution (float)
        :points: number of points in displacement grid -> points * points (int)
        :interpolation_magnitude: magnitude of interpolation, no interpolation if 0 ({0, 1, 2, 3, 4}) 
        :mode: border mode (({nearest, wrap, reflect, mirror, constant}))
    '''
    image_array = np.asarray(x.data)[0]
    cval = 0.0
    prefilter = True
    axis = None
    if mode == "constant":
        cval = np.unique(image_array)[0]

    # prepare inputs and axis selection
    Xs = _normalize_inputs(image_array)
    axis, deform_shape = _normalize_axis_list(axis, Xs)

    if not isinstance(points, (list, tuple)):
        points = [points] * len(deform_shape)

    np.random.seed(seed=seed)
    displacement = np.random.randn(len(deform_shape), *points) * sigma

    image_array = elasticdeform.deform_grid(image_array, displacement=displacement,
                                            order=interpolation_magnitude, mode=mode, cval=cval, prefilter=prefilter, axis=axis)

    return pil2tensor(image_array, np.float32)


# wrapper for _elastic_transform
elastic_transform = TfmPixel(_elastic_transform)


# set random position range for random_crop and tile_shape
rand_pos_set = {'row_pct': (0.3, 0.7), 'col_pct': (0.3, 0.7)}
tile_shape_set = (540, 540)


def get_custom_transforms(do_flip: bool = True,
                          flip_vert: bool = True,
                          elastic_deformation: bool = True,
                          elastic_deform_args={'sigma': 10, 'points': 10},
                          random_crop=tile_shape_set,
                          rand_pos=rand_pos_set,
                          max_rotate: float = 10.,
                          max_lighting: float = 0.2,
                          p_affine: float = 0.75,
                          p_lighting: float = 0.75,
                          p_deformation=0.9,
                          transform_valid_ds=False,
                          xtra_tfms: Optional[Collection[Transform]] = None) -> Collection[Transform]:
    '''
    Utility func to easily create a list of random_crop, flip, rotate, elastic deformation and lighting transforms.
        :param do_flip: apply random flip (bool)
        :param flip_vert: additionally apply vertical flip, if do_flip = True (bool)
        :param elastic_transform: apply elastic_deformation (bool)
        :param elastic_deform_args: arguments passed into elastic_deformation (dict)
        :param random_crop: tile_shape of random crop (tuple or None)
        :param max_rotate: max angle for random rotation (float or None)
        :param max_lighting: max lighting increase (float or None)
        :param p_affine: probability for affine transformations (float)
        :param p_lighting: probability for lighting transformations (float)
        :param p_deformation: probability for deformation (float)
        :param transform_valid_ds: apply transforms to validation dataset (bool)

        :return: List of transforms for train_ds and valid_ds

    '''
    res = []
    if random_crop:
        res.append(initial_crop_pad(
            height=random_crop[0], width=random_crop[1], **rand_pos))
    if do_flip:
        res.append(dihedral_affine() if flip_vert else flip_lr(p=0.5))
    if max_rotate:
        res.append(rotate(degrees=(-max_rotate, max_rotate), p=p_affine))
    if elastic_deformation:
        res.append(elastic_transform(p=p_deformation,
                                     seed=(0, 100000), **elastic_deform_args))
    if max_lighting:
        res.append(brightness(change=(0.5, 0.5*(1+max_lighting)),
                              p=p_lighting, use_on_y=False))
        res.append(contrast(scale=(1, 1/(1-max_lighting)),
                            p=p_lighting, use_on_y=False))

    # either transform valid_ds too or just add random crop
    if transform_valid_ds:
        res_y = res
    else:
        if random_crop:
            res_y = [initial_crop_pad(
                height=random_crop[0], width=random_crop[1], **rand_pos)]
        else:
            res_y = []

    #       train                   , valid
    return (res + listify(xtra_tfms), res_y)


"""CUSTOM ItemBase"""


class WeightedLabels(ItemBase):
    '''
    Custom ItemBase to store and process labels and pixelwise weights together.
    Also handling the target_size of the masks.
      :param lbl:          Image file of a label
      :param wgt:          Image file of a weightmap
      :param target_size:  dimensions of the network's output size
    '''

    def __init__(self, lbl:Image, wgt:Image, target_size:tuple = None):
        self.lbl, self.wgt = lbl, wgt
        self.obj, self.data = (lbl, wgt), [lbl.data, wgt.data]

        self.target_size = target_size

    def apply_tfms(self, tfms, **kwargs):
        # if mask should be cropped, add operation 'do_crop_y' to transforms
        crop_to_target_size = self.target_size
        
        if not tfms:
          tfms = []
        
        # Change interpolation of elastic deformation to 0
        for tfm in tfms:
          if 'interpolation_magnitude' in tfm.resolved.keys():
            tfm.resolved['interpolation_magnitude'] = 0
        
        self.lbl = self.lbl.apply_tfms(tfms, mode='nearest', **kwargs)
        self.wgt = self.wgt.apply_tfms(tfms, mode='nearest', **kwargs)
        
        if crop_to_target_size:
          self.lbl = self.lbl.crop(crop_to_target_size)
          self.wgt = self.wgt.crop(crop_to_target_size)
        
        self.data = [self.lbl.data, self.wgt.data]
        return self

    def __repr__(self):
        return f'{self.__class__.__name__}{(self.lbl, self.wgt)}'


"""CUSTOM ItemList AND ItemLabelList"""


class CustomSegmentationLabelList(ImageList):
    "'Item List' suitable for WeightedLabels containing labels and pixelweights"
    _processor = vision.data.SegmentationProcessor

    def __init__(self,
                 items: Iterator,
                 wghts = None,
                 classes: Collection = None,
                 target_size: Tuple = None,
                 loss_func=CrossEntropyFlat(axis=1),
                 **kwargs):

        super().__init__(items, **kwargs)
        self.copy_new.append('classes')
        self.copy_new.append('wghts')
        self.classes, self.loss_func, self.wghts = classes, loss_func, wghts
        self.target_size = target_size

    def open(self, fn):
        res = io.imread(fn)
        res = pil2tensor(res, np.float32)
        return Image(res)

    def get(self, i):
        fn = super().get(i)
        wt = self.wghts[i]
        return WeightedLabels(fn, self.open(wt), self.target_size)

    def reconstruct(self, t: Tensor):
        return WeightedLabels(Image(t[0]), Image(t[1]), self.target_size)

    #def analyze_pred(self, pred, thresh:float=0.5): return pred.argmax(dim=0)[None]


class CustomSegmentationItemList(ImageList):
    "'ItemList' suitable for segmentation with pixelwise weighted loss"
    _label_cls, _square_show_res = CustomSegmentationLabelList, False

    def label_from_funcs(self, get_labels: Callable, get_weights: Callable,
                         label_cls: Callable = None, classes=None,
                         target_size: Tuple = None, **kwargs) -> 'LabelList':
        "Get weights and labels from two functions. Saves them in a CustomSegmentationLabelList"
        kwargs = {}
        wghts = [get_weights(o) for o in self.items]
        labels = [get_labels(o) for o in self.items]

        if target_size:
            print(
                f'Masks will be cropped to {target_size}. Choose \'target_size = None\' to keep initial size.')
        else:
            print(f'Masks will not be cropped.')

        y = CustomSegmentationLabelList(
            labels, wghts, classes, target_size, path=self.path)
        res = self._label_list(x=self, y=y)
        return res

    def show_xys(self, xs, ys, figsize: Tuple[int, int] = None, padding=184, **kwargs):
        "Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."

        if not figsize:
            figsize = (18, 6*len(xs))

        # colormap for labels
        lmap = plt.get_cmap('Dark2')
        lmap.set_under('k', alpha=0)

        # colormap for weights
        wmap = plt.get_cmap('YlOrRd')  # YlOrRd PuRd
        wmap.set_under('k', alpha=0)
        wmax = np.unique(ys[0].wgt.data)[-1]*0.9

        same_size = (ys[0].lbl.size == xs[0].size)

        # get width from x and set red boundingbox
        wdt, hgt = xs[0].size

        if not same_size:
            padding = int(0.5*(wdt - ys[0].lbl.size[1]))  # numpy pad
        else:
            padding = int(0.5*padding)

               #(x-offset,y-offset)      ,width      ,height
        bbox = patches.Rectangle((padding-1, padding-2), wdt+2-2*padding,
                                 hgt+2-2*padding, edgecolor='r', linewidth=1, facecolor='none')
        
        # rows: number of batch items
        rows = len(xs)
        fig, axs = plt.subplots(rows, 3, figsize=figsize)
        # idx: batch item counter
        idx = 0
        for i, ax in enumerate(axs.flatten() if rows > 1 else axs):
            # display image and labels together
            if(i % 3 == 0):
                # normalize the images for a better visualization
                imax = np.unique(xs[idx].data)[-1]
                imax = 0.8
                
                y_label = np.asarray(ys[idx].lbl.data[0])

                if not same_size:
                    y_label = np.pad(y_label, padding,
                                     'constant', constant_values=(0))


                ax.imshow(np.asarray(xs[idx].data[0]), vmin=0, vmax=imax,
                          cmap='binary', interpolation='none')  # cmap= bone_r
                ax.imshow(y_label, cmap=lmap, vmin=0.2, vmax=1,
                          alpha=0.8, interpolation='none')
                                
                ax.add_patch(copy(bbox))
                ax.set_title('Image + Labels')

            # display image and weights together
            elif(i % 3 == 1):

                y_weights = np.asarray(ys[idx].wgt.data[0])

                if not same_size:
                    y_weights = np.pad(y_weights, padding,
                                       'constant', constant_values=(0))

                ax.imshow(np.asarray(xs[idx].data[0]),vmin=0, vmax=imax,
                          cmap='binary', interpolation='none')
                ax.imshow(y_weights,vmin=0.12, vmax=wmax, cmap=wmap,
                           alpha=0.5, interpolation='none')

                ax.add_patch(copy(bbox))
                ax.set_title('Image + Weights')

            # display labels and weights together
            else:
                #weights
                ax.imshow(y_weights, vmin=0.12, vmax=wmax, cmap=wmap,
                           interpolation='none')
                #labels
                ax.imshow(y_label, cmap=lmap, vmin=0.1, vmax=1,
                          alpha=1, interpolation='none')
                #patch
                ax.add_patch(copy(bbox))
                ax.set_title('Labels + Weights')

                idx += 1

        plt.tight_layout()

    def show_xyzs(self, xs, ys, figsize: Tuple[int, int] = None, padding=184, **kwargs):
        "Show the `xs`, `ys`and `zs` on a figure of `figsize`. `kwargs` are passed to the show method."

        if not figsize:
            figsize = (18, 6*len(xs))

        # colormap for labels
        lmap = plt.get_cmap('Blues')
        lmap.set_under('k', alpha=0)

        pmap = plt.get_cmap('Reds')
        pmap.set_under('k', alpha=0)

        same_size = (ys[0].lbl.size == xs[0].size)

        # get width from x and set red boundingbox
        wdt, hgt = xs[0].size

        if not same_size:
            padding = int(0.5*(wdt - ys[0].lbl.size[1]))  # numpy pad
        else:
            padding = int(0.5*padding)

            #(x-offset,y-offset)      ,width      ,height
        bbox = patches.Rectangle((padding-1, padding-1), wdt-2*padding,
                                 hgt-2*padding, edgecolor='r', linewidth=1, facecolor='none')

        # rows: number of batch items
        rows = len(xs)
        fig, axs = plt.subplots(rows, 3, figsize=figsize)
        # idx: batch item counter
        idx = 0
        for i, ax in enumerate(axs.flatten() if rows > 1 else axs):

            # display image and labels together
            if(i % 3 == 0):
                # normalize the images for a better visualization
                imax = np.unique(xs[idx].data)[-1]

                y_label = np.asarray(ys[idx].lbl.data[0])

                if not same_size:
                    y_label = np.pad(y_label, padding,
                                     'constant', constant_values=(0))

                ax.imshow(np.asarray(xs[idx].data[0]), vmax=imax,
                          cmap='binary', interpolation='none')  # cmap= bone_r
                ax.imshow(y_label, cmap=lmap, vmin=0.2,
                          alpha=0.8, interpolation='none')
                ax.add_patch(copy(bbox))
                ax.set_title('Image + Labels')

            # display image and predictions together
            elif(i % 3 == 1):

                y_predictions = np.asarray(zs[idx].data[0])

                if not same_size:
                    y_predictions = np.pad(
                        y_predictions, padding, 'constant', constant_values=(0))

                ax.imshow(np.asarray(xs[idx].data[0]), vmax=imax,
                          cmap='binary', interpolation='none')
                ax.imshow(y_predictions, vmax=wmax, cmap=wmap,
                          vmin=0.1, alpha=0.8, interpolation='none')

                ax.add_patch(copy(bbox))
                ax.set_title('Image + Predictions')

            # display labels and weights together
            else:
                ax.imshow(y_predictions, vmax=wmax, cmap=wmap,
                          vmin=0.1, alpha=0.6, interpolation='none')
                ax.imshow(y_label, cmap=lmap, vmin=0.1,
                          alpha=0.6, interpolation='none')
                ax.add_patch(copy(bbox))
                ax.set_title('Labels + Predictions')

                idx += 1
        plt.tight_layout()


"""CUSTOM LOSS"""

class FlattenedWeightedLoss():
    "Same as `func`, but flattens input and target."

    def __init__(self, func, *args,
                 axis: int = 1,
                 reduction_mode='sum',
                 longify: bool = True,
                 is_2d: bool = True,
                 **kwargs):
        self.func, self.axis, self.longify, self.is_2d = func(
            *args, **kwargs), axis, longify, is_2d
        self.reduction_mode = reduction_mode

    def __repr__(self): return "WeightedCrossEntropyLoss"
    @property
    def reduction(self): return self.func.reduction
    @reduction.setter
    def reduction(self, v): self.func.reduction = v

    def __call__(self,
                 input: Tensor,
                 labels: Tensor,
                 weights: Tensor,
                 **kwargs) -> Rank0Tensor:

        assert self.reduction_mode in ('sum', 'mean', 'none'), \
            'Check reduction_mode and chose between sum, mean or none'

        # flatten
        input = input.transpose(self.axis, -1).contiguous()
        labels = labels.transpose(self.axis, -1).contiguous()
        weights = weights.transpose(self.axis, -1).contiguous()

        # transform to long
        if self.longify:
            labels = labels.long()

        # reshape
        input = input.view(-1, input.shape[-1]
                           ) if self.is_2d else input.view(-1)
        labels = labels.view(-1)
        weights = weights.view(-1)

        res = nn.CrossEntropyLoss(reduction='none')(input, labels, **kwargs)

        if self.reduction_mode == 'sum':
            return (weights * res).sum()
        elif self.reduction_mode == 'mean':
            return (weights * res).mean()
        else:
            return weights * res


def WeightedCrossEntropyLoss(*args, axis: int = -1, **kwargs):
    "Same as `nn.CrossEntropyLoss`, but flattens input and target."
    return FlattenedWeightedLoss(nn.CrossEntropyLoss, *args, axis=axis, **kwargs)


"""METRICS"""

class CustomMetrics(Callback):
    '''
    Wrap a metric `func` in a callback for metrics computation. Gets rid of weights, 
    decouples tensors from device and changes name of function to be displayed during training.    
      :param func: a functions that takes preds and targets (in this order) to compute a quality measure
      :param name: name to be displayed (takes func name if not specified)
      :param swap_preds: swap prediction with targets
      :param **kwargs: additional arguments are past into func
      
      :returns: average metric from last epoch
    '''
    def __init__(self, func, name= None, swap_preds= False, **kwargs):
      
        if not name:
          name = getattr(func,'func',func).__name__
        self.func, self.name = func, name
        self.swap_preds = swap_preds
        self.kwargs = kwargs

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = 0.,0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        last_target = last_target[0]

        #if not is_listy(last_target): last_target=[last_target]
        this_size = last_target.size(0)
        self.count += this_size

        pred = last_output.argmax(dim=1).flatten().cpu()
        targs = last_target.flatten().cpu()
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          
          if self.swap_preds:
            val = self.func(targs, pred,**self.kwargs)
          else:
            val = self.func(pred, targs,**self.kwargs)
            
        self.val += this_size * val

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, self.val/self.count)
        
        
"""INITIATE LEANER"""

class initiate_learn():
  '''
  Class to initiate the learner during training. 
    :param model_dir:  Path to the model directory
    :param net:        nn.torch model
    :param metrics:    list of metrics
    :param loss_func:  loss function
    :param opt_func:   optimization function
    :param csv_log:    True or False to save logs of loss and metrics as additional CSV
  '''

  def __init__(self, model_dir, network, metrics, loss_func, opt_func = torch.optim.Adam, csv_log:bool=True):
    self.model_dir = model_dir
    self.net = network
    self.metrics = metrics
    self.opt_func = opt_func
    self.csv_log = csv_log
    self.loss_func = loss_func
  
  def __call__(self, data, cycle:int=0, wd = 10e-02, csv_name:str = 'metrics',early_stopping=None,patience=5):
    '''
    Learner object is initiated on call by handing over data and cycle count to change the directory to cycle
      :param data:           data object
      :param cycle:          cycle count
      :param wd:             weight decay
      :param csv_name:       csv_name to save metrics
      :param early_stopping: metric for early stopping monitoring
      :param patience:       tell early stopping to wait for improvement for this amount of epochs 
      
      :return:  returns learner object
    '''    
    cycle_dir = f'{self.model_dir}/{cycle}'
    
    callbacks_list = [ShowGraph]
    
    if self.csv_log:
      callbacks_list.append(partial(CSVLogger, filename=f'{self.model_dir}/{cycle}/{csv_name}'))
      
    if early_stopping:
      callbacks_list.append(partial(EarlyStoppingCallback, monitor='mcc', min_delta=0.01, patience=patience))
    
    learn = Learner(data,
                    self.net,
                    model_dir = self.model_dir,
                    opt_func = self.opt_func,
                    metrics = self.metrics,
                    loss_func = self.loss_func,
                    wd = wd,
                    callback_fns=callbacks_list)
                    
    if not os.path.exists(cycle_dir):
      os.makedirs(cycle_dir)
      print(f'Cycle {cycle} directory created.')

    return learn


"""Tile Generator"""
class TileGenerator:
  '''
  Tile generator to split input images into multiple tiles, to be used by learner. 
  Images can be padded to get a prediction of complete image.
    :param data:          Listify object of images or DataLoader
    :param learner:       The loaded fastai learner model
    :param tile_shape:    Shape of the network input
    :param mask_shape:    Shape of the network output 
    :param same_size:     True if all input images are the same size
    :param padding_mode:  Set padding to analyze full image, chose between [None, 'zeros', 'border' or 'reflection']
  '''
  def __init__(
      self,
      data,
      learner,
      tile_shape=(540,540),
      mask_shape=(356,356),
      same_size = True,
      padding_mode = 'reflection' 
  ):
    
    self.data = [image for image in tqdm(data,desc='Loading Data: ')]

    self.learner = learner
    self.tile_shape = tile_shape
    self.mask_shape = mask_shape
    self.same_size = same_size
    
    # pad all images with selected mode
    if padding_mode:
      padding = tuple(np.subtract(tile_shape,mask_shape)//2)
      self.pad_images(padding, padding_mode)
    
    # get input image shapes
    self.data_shapes = [self.data[0].shape]*len(data) if same_size else [item.shape for item in self.data]

    # empty initialisation for tiles and predictions
    self.tiles = None
    self.predictions = None

    self.tile_dimensions = None

    # split tiles upon initiation
    self.split_tiles()

  def __len__(self):
      return len(self.data)

  def __getitem__(self, index):
      if self.tiles == None:
        return self.data[index]
      else:
        return self.tiles[index]

  def tile_splitter(self,input_shape:tuple):
    '''
    Gets the pixelwise regions of the tiles for cropping the image. Tiles need 
    to overlap, if your model doesn't use padding to compensate the cropping of 
    Convolutions.
    
      :param input_shape: Input shape of the Image in form of (C,H,W)

      :return tiles:      pixelwise tile areas (xs,xf,ys,yf) with S(xs,ys) being left 
                          upper corner and F(xf,yf) being the lower right corner of our rectangular tile
      :return tile_dimensions: decimal numbers of how many full and cropped tiles are needed for given
                               input image shape in form of (xtiles,ytiles)
    '''
    
    tx, ty = self.tile_shape
    mx, my = self.mask_shape
    px, py = np.subtract((tx,ty),(mx,my))

    _, dy, dx = input_shape

    # how many tiles are needed (decimal precision)
    xtiles = (dx-px)/mx 
    ytiles = (dy-py)/my 

    # add starting points for full tiles with spacing mx and my
    x_start = [0+mx*ix for ix in range(int(xtiles))]
    y_start = [0+my*iy for iy in range(int(ytiles))]

    # add a last x, y starting point for non-integer tiles in xtiles,ytiles
    if not xtiles%1==0:
      x_start.append(dx-tx)

    if not ytiles%1==0:
      y_start.append(dy-ty)

    # build tiles with width tx and height ty
    tiles = []
    for y in y_start:
      for x in x_start:
        tiles.append((x,x+tx,y,y+ty))

    tile_dimensions = (xtiles,ytiles)
    
    return tiles, tile_dimensions

  def split_tiles(self):
    '''
    Splits the images into tiles and saves them to 'self.tiles' as well as the float 
    precision of the needed amount of tiles into 'self.tile_dimensions' in form of 
    a list of tuples (xtiles,ytiles)
    '''
    ds = self.data_shapes

    # if all input images have the same shape
    if self.same_size:
      
      # call tile_splitter to split all images depending on their shape
      tile_regions, tile_dimension =  self.tile_splitter(ds[0])
      tiles = []

      for img in tqdm(self.data,desc='Building Tiles'):
        img_tiles = []
        for region in tile_regions:
          img_tiles.append(self.crop_to_tile(img,region))
        tiles.append(img_tiles)

      self.tiles = tiles
      self.tile_dimensions = [tile_dimension]*len(self.data)
    
    #if input images have different shapes
    else:
      
      tiles = []
      tile_dimensions =[]
      
      # cycle through all images
      for i, img in tqdm(enumerate(self.data),desc='Building Tiles'):
        
        # call tile_splitter to split image depending on image shape
        tile_regions, tile_dimension =  self.tile_splitter(ds[i])
        img_tiles = []

        for region in tile_regions:
          img_tiles.append(self.crop_to_tile(img, region))

        tiles.append(img_tiles)
        tile_dimensions.append(tile_dimension)

      self.tiles = tiles
      self.tile_dimensions = tile_dimensions

  def crop_to_tile(self, img:Image, region):
    '''
    Crops input image to rectangular region from corner (xs,ys) to corner (xf,yf)
      :param img: 	The Image to be cropped
      :param region: 	Tuple defining pixels for cropped region in form (xs,xf,ys,yf),
              s denoting the starting pixels and f denoting the final pixels

      :return: 		Cropped Image
    '''
    
    xs,xf,ys,yf = region
    return Image(img.data[:,ys:yf,xs:xf])
  
  def pad_images(self,padding,padding_mode): 
    '''
    Pads all images in self.data 
      :param padding: Magnitude of padding
    '''
    assert padding[0] == padding[1], 'For padding: tile_shape and mask_shape need to be squares.'
    self.data = [image.pad(padding[0],padding_mode) for image in tqdm(self.data,desc='Padding images')]

  def display_tiles(self,batch:Iterator,shape:tuple=(3,3),figsize:tuple=(9,9)):
    '''
    Displays given tiles in 'batch' in subplots arranged by 'shape' within figure
    with figsize of 'figsize'
      :param batch: 	List of Images
      :param shape: 	Rows and colums of subplots (tuple)
      :param figsize: Figsize of plt figure
    '''

    # get heights and width of all columns and rows
    widths= [batch[i].shape[2] for i in range(shape[0])]
    heights = [batch[shape[1]*i].shape[1] for i in range(shape[1])]

    specs = dict(width_ratios=widths, height_ratios=heights, wspace=0.04, hspace=0.04)

    f, axarr = plt.subplots(shape[0], shape[1], 
                            figsize=figsize, 
                            gridspec_kw = specs
                            )

    for ax,tile in zip(axarr.flatten(), batch):
      ax.set_xticklabels([])
      ax.set_yticklabels([])

      tile.show(ax)

  def stitch_helper(self,dimensions,mask_shape,reshape = True):
    '''
    creates a lookup matrix on how the output tiles need to be cropped for stitching.
      :param dimensions: 	How many full and partial tiles are needed for concatenation in float precision (N,M)
      :param mask_shape: 	The shape of the prediction output (W,H)
      :param reshap: 		  Reshape to an indexable 1D array with length ceil(N)xceil(M)

      :return: 			      Lookup matrix (np.array (NxM,R) or (N,M,R)) with R being the crop region (xs,xf,ys,yf) for tile (N,M)
    '''
    xd,yd = dimensions
    xm,ym = mask_shape
    rows  = ceil(xd)*ceil(yd)
    lookup_matrix = np.ones((ceil(yd),ceil(xd),4))*(0,xm,0,ym)
    
    # for all columns in tiles
    for ix in range(ceil(xd)):
      xd -= 1
      
      # if no full tile needed, compute xs
      if xd<0:
        lookup_matrix[:,ix,0] = -xd*xm
      else:
        lookup_matrix[:,ix,0] = 0
    
    # for all rows in tiles
    for iy in range(ceil(yd)):
      yd -= 1
      
      # if no full tile needed, compute ys
      if yd<0:
        lookup_matrix[iy,:,2] = -yd*ym
      else:
        lookup_matrix[iy,:,2] = 0
    
    # return
    if reshape:
      return np.reshape(lookup_matrix,(rows,4)).astype(int)
    else:
      return lookup_matrix.astype(int)

  def show_tiles(self, index, crop_to = 'mask', base_size=3):
    '''
    Displays tiles on a set of subplots.
      :param index: 	Index of instance to be displayed (int)
      :param crop_to: Crop is either None, 'mask' or 'original' {None,'mask','original'}
      :base_size: 	  Base size for each tile
    '''
    
    #deepcopy to not alter the originals
    batch = deepcopy(self.tiles[index])
    dimensions = self.tile_dimensions[index]
    x, y = np.ceil(dimensions).astype(int)
    
    # number of tiles to display, adapt figsize to number
    shape = (x,y)
    figsize = (x*base_size,y*base_size)

    # crop to mask shape
    if crop_to == 'mask':
      for tile in batch:
        tile = tile.crop(self.mask_shape)
    
    # crop to input image with lookup_matrix
    elif crop_to == 'original':  
      lookup_matrix = self.stitch_helper(dimensions,self.mask_shape)  

      # crop each tile accordingly      
      for i, (region, tile) in enumerate(zip(lookup_matrix,batch)):

        tile = tile.crop(self.mask_shape) 
        batch[i] = self.crop_to_tile(tile,region)

    self.display_tiles(batch,shape,figsize)

  def show_predictions(self):
    print('Not implemented')

  def predict_instance(self, x:Image):
    '''
    Calls FastAIs Learner.predict() function and transforms it into a prediction 
    mask.
      :param x: Tile to be predicted (Image)

      :return: 	Prediction (Image)
      :return: 	Probabilities (Tensor)
    '''
    raw = self.learner.predict(x)
    probabilities, values = raw[1].max(0)

    return Image(values.unsqueeze(0)), probabilities.unsqueeze(0)

  def predict_all(self):
    '''
    Predicts all tiles from all images by calling predict_instance() and saves 
    them together with the pixelwise probabilities to self.predictions.
    '''
    solutions = []
    probabilities = []
    
    # iterate all images
    for batch in tqdm(self.tiles,desc='Predicting Tiles'):
      batch_solutions = []
      batch_probabilities = []

      # iterate all tiles in image and predict
      for tile in batch:

        tile_prediction, tile_probabilities = self.predict_instance(tile)
        batch_solutions.append(tile_prediction)
        batch_probabilities.append(tile_probabilities)

      # append imagewise solutions and probabilities 
      solutions.append(batch_solutions)
      probabilities.append(batch_probabilities)

    self.predictions = [solutions,probabilities]
  
  def stitch_image(self,tile_list,lookup_matrix,is_img=True):
    '''
    Stitch image by concatenating and cropping tiles, depending on lookup_matrix
      :param tile_list: 		Array of tiles to form an image (M,N)
      :param lookup_matrix: 	Array values to crop the images

      :return: 				Concatenated Image 
    '''
    image = None
    
    for brow,lmrow in zip(tile_list, lookup_matrix):

      row = None
      for tile, region in zip(brow,lmrow):
        
        # crops tile region if region of interest is smaller than given input
        if not np.array_equal(region,[0,self.mask_shape[0],0,self.mask_shape[1]]):
          if is_img:
            tile = self.crop_to_tile(tile, region)
          else:
            xs,xf,ys,yf = region
            tile = tile[:,ys:yf,xs:xf]
        
        # concat columns
        if row is None:
          row = tile.data if is_img else tile
        else:
          row = torch.cat((row,tile.data if is_img else tile),2)

      # concat rows
      if image is None:
        image = row
      else:
        image = torch.cat((image,row),1)
    
    return Image(image) if is_img else image

  def stitch_results(self):
    '''
    Stitches the results in self.predictions and returns them to the user
      :return:	List with Images and Probability maps
    '''
    xs,ys = self.mask_shape
    results = []
    probs = []

    # cycle through predicted labels and their probabilities
    for tiles, probabilities, dimensions in zip(self.predictions[0],
                                                self.predictions[1],
                                                self.tile_dimensions):
     
      # retrieve lookup_matrix on how to cut results for stitching
      xd,yd = np.ceil(dimensions)
      lookup_matrix = self.stitch_helper(dimensions,(xs,ys),reshape=False) 
      _reshape = lookup_matrix.shape[0:2]
      
      # convert probabilities to images and arrange in 2D array
      tiles_array = np.reshape(tiles,_reshape) 
      prob_array = np.reshape([Image(p) for p in probabilities],_reshape)
      
      # stitch images and probabilities back together
      image = self.stitch_image(tiles_array,lookup_matrix)
      prob_map = self.stitch_image(prob_array,lookup_matrix)


      results.append(image)
      probs.append(prob_map)
    
    # crop original images
    [image.crop(self.data_shapes[i]) for i, image in enumerate(self.data)]

    return results, probs