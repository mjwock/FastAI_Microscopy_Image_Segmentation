import torch
from . import _functions
from .modules import utils
from ._functions.padding import ConstantPad2d
from .modules.utils import _single, _pair, _triple

def log_softmax(input):
	return _functions.thnn.LogSoftmax()(input)
	
def nll_loss(input, target, weight=None, size_average=True):
	r"""The negative log likelihood loss.
	See :class:`~torch.nn.NLLLoss` for details.
	Args:
		input: :math:`(N, C)` where `C = number of classes`
		target: :math:`(N)` where each value is `0 <= targets[i] <= C-1`
		weight (Variable, optional): a manual rescaling weight given to each
				class. If given, has to be a Variable of size "nclasses"
		size_average (bool, optional): By default, the losses are averaged
				over observations for each minibatch. However, if the field
				sizeAverage is set to False, the losses are instead summed
				for each minibatch.
	Attributes:
		weight: the class-weights given as input to the constructor
	Example:
		>>> # input is of size nBatch x nClasses = 3 x 5
		>>> input = autograd.Variable(torch.randn(3, 5))
		>>> # each element in target has to have 0 <= value < nclasses
		>>> target = autograd.Variable(torch.LongTensor([1, 0, 4]))
		>>> output = F.nll_loss(F.log_softmax(input), target)
		>>> output.backward()
	"""
	dim = input.dim()
	if dim == 2:
		f = _functions.thnn.NLLLoss(size_average, weight=weight)
	elif dim == 4:
		f = _functions.thnn.NLLLoss2d(size_average, weight=weight)
	else:
		raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))
	return f(input, target)

def cross_entropy(input, target, weight=None, size_average=True):
	"""This criterion combines `log_softmax` and `nll_loss` in one single class.
	See :class:`torch.nn.CrossEntropyLoss` for details.
	Args:
		input: Variable :math:`(N, C)` where `C = number of classes`
		target: Variable :math:`(N)` where each value is `0 <= targets[i] <= C-1`
		weight (Tensor, optional): a manual rescaling weight given to each
				class. If given, has to be a Tensor of size "nclasses"
		size_average (bool, optional): By default, the losses are averaged
				over observations for each minibatch. However, if the field
				sizeAverage is set to False, the losses are instead summed
				for each minibatch.
	"""
	return nll_loss(log_softmax(input), target, weight, size_average)

def dice_loss(input,target):
	"""
	input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
	target is a 1-hot representation of the groundtruth, shoud have same size as the input
	"""
	assert input.size() == target.size(), "Input sizes must be equal."
	assert input.dim() == 4, "Input must be a 4D Tensor."
	uniques=np.unique(target.numpy())
	assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

	probs=F.softmax(input)
	num=probs*target#b,c,h,w--p*g
	num=torch.sum(num,dim=3)#b,c,h
	num=torch.sum(num,dim=2)
	

	den1=probs*probs#--p^2
	den1=torch.sum(den1,dim=3)#b,c,h
	den1=torch.sum(den1,dim=2)
	

	den2=target*target#--g^2
	den2=torch.sum(den2,dim=3)#b,c,h
	den2=torch.sum(den2,dim=2)#b,c
	

	dice=2*(num/(den1+den2))
	dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

	dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

	return dice_total
	
def weighted_dice_cross_entropy_loss(input,target, cross_entropy_weight = 0.5, dice_entropy_weight = 0.5):
	"""
	This function combines cross entropy loss and dice overlap coefficient loss in a weighted manner.
	Args:
		cross_entropy_weight: weight factor for the cross entropy loss
		dice_entropy_weight: weight factor for the dice overlap coefficient loss
	"""
	return cross_entropy_weight * cross_entropy(input,target) + dice_entropy_weight * dice_loss(input,target)
	