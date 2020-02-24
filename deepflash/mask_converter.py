from fastai.vision import *
from fastai.callbacks.hooks import *
import PIL.Image as PilImage

def getClassValues(label_names):

	containedValues = set([])

	for i in range(len(label_names)):
		tmp = open_mask(label_names[i])
		tmp = tmp.data.numpy().flatten()
		tmp = set(tmp)
		containedValues = containedValues.union(tmp)
	
	return list(containedValues)

def replaceMaskValuesFromZeroToN(mask, 
								 containedValues):

	numberOfClasses = len(containedValues)
	newMask = np.zeros(mask.shape)

	for i in range(numberOfClasses):
		newMask[mask == containedValues[i]] = i
	
	return newMask

def convertMaskToPilAndSave(mask, 
							saveTo):

	imageSize = mask.squeeze().shape

	im = PilImage.new('L',(imageSize[1],imageSize[0]))
	im.putdata(mask.astype('uint8').ravel())
	im.save(saveTo)

def convertMasksToGrayscaleZeroToN(pathToLabels,
								   saveToPath):

	label_names = get_image_files(pathToLabels)
	containedValues = getClassValues(label_names)

	for currentFile in label_names:
		currentMask = open_mask(currentFile).data.numpy()
		convertedMask = replaceMaskValuesFromZeroToN(currentMask, containedValues)
		convertMaskToPilAndSave(convertedMask, f'{saveToPath}/{currentFile.name}')
	
	print('Conversion finished!')