from fastai.vision import *
from skimage import io
import matplotlib.pyplot as plt
import random
import warnings


def show_example_data_batch(image_path, img_df:DataFrame, get_labels:Callable, get_weights:Callable,cmap='viridis', n:int=3):
	
	assert n>0, 'n must be greater than 0.'
	
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
			axarr[0].title.set_text('Image')
			axarr[1].imshow(maskEx,cmap=cmap[1])
			axarr[1].title.set_text('Labels')
			axarr[2].imshow(weightEx, cmap=cmap[2])
			axarr[2].title.set_text('Weights')

