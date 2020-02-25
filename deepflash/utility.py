from fastai.vision import *
from skimage import io
import matplotlib.pyplot as plt
import random


def show_example_data_batch(image_path, img_df:DataFrame, get_labels:Callable, get_weights:Callable, n:int=3):
	rndImages = img_df.sample(n)
	rndImages = rndImages.to_numpy()

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
		axarr[0].imshow(imgEx)
		axarr[0].title.set_text('Image')
		axarr[1].imshow(maskEx)
		axarr[1].title.set_text('Labels')
		axarr[2].imshow(weightEx)
		axarr[2].title.set_text('Weights')

