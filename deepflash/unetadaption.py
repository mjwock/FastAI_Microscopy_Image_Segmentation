import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
	def __init__(
		self,
		in_channels=1,
		n_classes=2,
		depth=5,
		wf=6,
		padding=False,
		batch_norm=False,
		dropout=None,
		up_mode='upconv',
		leaky_slope =0.1
	):
		"""
		Implementation of
		U-Net: Convolutional Networks for Biomedical Image Segmentation
		(Ronneberger et al., 2015)
		https://arxiv.org/abs/1505.04597
		Using the default arguments will yield the exact version used
		in the original paper
		Args:
			in_channels (int): number of input channels
			n_classes (int): number of output channels
			depth (int): depth of the network
			wf (int): number of filters in the first layer is 2**wf
			padding (bool): if True, apply padding such that the input shape
							is the same as the output.
							This may introduce artifacts
			batch_norm (bool): Use BatchNorm after layers with an
							   activation function
			up_mode (str): one of 'upconv' or 'upsample'.
						   'upconv' will use transposed convolutions for
						   learned upsampling.
						   'upsample' will use bilinear upsampling.
			leaky_slope (float): the slope of the leaky ReLU
			dropout (float): dropout probability for results after BatchNorm layers
		"""
		super(UNet, self).__init__()
		assert up_mode in ('upconv', 'upsample')
		self.padding = padding
		self.depth = depth
		prev_channels = in_channels
		self.down_path = nn.ModuleList()
		for i in range(depth):
			if i == 0:
				self.down_path.append(
					UNetConvBlock(prev_channels, 2 ** (wf + i), padding, False, leaky_slope, False)
				)
			else:
				self.down_path.append(
					UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm, leaky_slope, dropout)
				)
			prev_channels = 2 ** (wf + i)

		self.up_path = nn.ModuleList()
		for i in reversed(range(depth - 1)):
			if i==0:
				self.up_path.append(
					LastUpBlock(prev_channels, prev_channels, wf, up_mode, padding, batch_norm, leaky_slope, dropout)
				)
				prev_channels = prev_channels
			else:
				self.up_path.append(
					UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm, leaky_slope, dropout)
				)
				prev_channels = 2 ** (wf + i)

		self.last = nn.ModuleList([nn.Conv2d(prev_channels, n_classes, kernel_size=1)])

	def forward(self, x):
		blocks = []
		for i, down in enumerate(self.down_path):
			x = down(x)
			if i != len(self.down_path) - 1:
				blocks.append(x)
				x = F.max_pool2d(x, 2)

		for i, up in enumerate(self.up_path):
			x = up(x, blocks[-i - 1])
		
		for last in self.last:
			x = last(x)
		
		return x


class UNetConvBlock(nn.Module):
	def __init__(self, in_size, out_size, padding, batch_norm, leaky_slope, dropout):
		super(UNetConvBlock, self).__init__()
		block = []

		block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
		block.append(nn.LeakyReLU(leaky_slope))
		if batch_norm:
			block.append(nn.BatchNorm2d(out_size))
		if dropout:
			block.append(nn.Dropout(p=dropout))

		block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
		block.append(nn.LeakyReLU(leaky_slope))
		if batch_norm:
			block.append(nn.BatchNorm2d(out_size))
		if dropout:
			block.append(nn.Dropout(p=dropout))

		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		return out


class UNetUpBlock(nn.Module):
	def __init__(self, in_size, out_size, up_mode, padding, batch_norm, leaky_slope, dropout):
		super(UNetUpBlock, self).__init__()
		if up_mode == 'upconv':
			self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
		elif up_mode == 'upsample':
			self.up = nn.Sequential(
				nn.Upsample(mode='bilinear', scale_factor=2),
				nn.Conv2d(in_size, out_size, kernel_size=1),
			)

		self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm, leaky_slope, dropout)

	def center_crop(self, layer, target_size):
		_, _, layer_height, layer_width = layer.size()
		diff_y = (layer_height - target_size[0]) // 2
		diff_x = (layer_width - target_size[1]) // 2
		return layer[
			:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
		]

	def forward(self, x, bridge):
		up = self.up(x)
		crop1 = self.center_crop(bridge, up.shape[2:])
		out = torch.cat([up, crop1], 1)
		out = self.conv_block(out)

		return out
		
class LastUpBlock(nn.Module):
	def __init__(self, in_size, out_size, wf, up_mode, padding, batch_norm, leaky_slope, dropout):
		super(LastUpBlock, self).__init__()
		if up_mode == 'upconv':
			self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
		elif up_mode == 'upsample':
			self.up = nn.Sequential(
				nn.Upsample(mode='bilinear', scale_factor=2),
				nn.Conv2d(in_size, out_size, kernel_size=1),
			)

		self.conv_block = UNetConvBlock(in_size + 2**wf, out_size, padding, batch_norm, leaky_slope, dropout)

	def center_crop(self, layer, target_size):
		_, _, layer_height, layer_width = layer.size()
		diff_y = (layer_height - target_size[0]) // 2
		diff_x = (layer_width - target_size[1]) // 2
		return layer[
			:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
		]

	def forward(self, x, bridge):
		up = self.up(x)
		crop1 = self.center_crop(bridge, up.shape[2:])
		out = torch.cat([up, crop1], 1)
		out = self.conv_block(out)

		return out