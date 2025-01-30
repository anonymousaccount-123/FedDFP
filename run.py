# from __future__ import print_function
# import argparse
# import os, sys
# from datetime import datetime
# import numpy as np
# import logging
# from utils.core_util import Meta
# import torch
#
#
# # Generic training settings
# parser = argparse.ArgumentParser(description='Configurations for Fed + DD + WSI Training')
# parser.add_argument('--repeat', type=int, default=5,
#                     help='number of repeated experiments')
# parser.add_argument('--data_root_dir', type=str, default=None,
#                     help='data directory')
# parser.add_argument('--global_epochs', type=int, default=200,
#                     help='maximum number of epochs to train globaly(default: 200)')
# parser.add_argument('--local_epochs', type=int, default=200,
#                     help='maximum number of epochs to train localy(default: 200)')
# parser.add_argument('--lr', type=float, default=1e-4,
#                     help='learning rate (default: 0.0001)')
# parser.add_argument('--reg', type=float, default=1e-5,
#                     help='weight decay (default: 1e-5)')
# parser.add_argument('--seed', type=int, default=1,
#                     help='random seed for reproducible experiment (default: 1)')
# parser.add_argument('--n_classes', type=int, default=2,
#                     help='number of classes')
# parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
# parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
# parser.add_argument('--best_run', type=int, default=0)
# parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
# parser.add_argument('--opt', type=str, default='adam')
# parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
# parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'mag'], default='ce',
#                      help='slide-level classification loss function (default: ce)')
# parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
# parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
# parser.add_argument('--model_size', type=str, choices=['tiny', 'ultra_small', 'small', 'big'], default='small', help='size of model, does not affect mil')
# parser.add_argument('--ft_model', type=str, default='ResNet50',
#                     choices=['ResNet50', 'ResNet50_prompt', 'ResNet50_deep_ft_prompt',
#                              'ResNet50_simclr', 'ResNet50_simclr_prompt',
#                              'ViT_S_16', 'ViT_S_16_prompt',
#                              'ViT_S_16_dino', 'ViT_S_16_dino_prompt', 'ViT_S_16_dino_deep_ft_prompt',
#                              'ViT_T_16', 'ViT_T_16_prompt', 'ViT_S_16_deep_ft_prompt', 'hipt'],)
# parser.add_argument('--mil_method', type=str, default='CLAM_SB', help='mil method')
# parser.add_argument('--fed_method', type=str, default='fed_avg',choices=['fed_avg',
#                                                                          'fed_prox',
#                                                                          'fed_dyn',
#                                                                          'scaffold',
#                                                                          'moon'], help='fed method')
# parser.add_argument('--mu', type=float, default=0.01, help='proximal term for fedprox')
# parser.add_argument('--alpha_coef', type=float, default=1e-2, help='alpha coefficient for feddyn')
# parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss for moon')
# parser.add_argument('--pool_option', type=str, default='FIFO', choices=['FIFO', 'LIFO'], help='pooling option for moon')
# parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
# parser.add_argument('--contrast_mu', type=float, default=1, help='the mu parameter for fedprox or moon')
# parser.add_argument('--task', type=str)
# parser.add_argument('--accumulate_grad_batches', type=int, default=1,)
# parser.add_argument('--use_h5', action='store_true', default=False, help='use h5 files')
# ### CLAM specific options
# parser.add_argument('--no_inst_cluster', action='store_true', default=False,
#                      help='disable instance-level clustering')
# parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
#                      help='instance-level clustering loss function (default: None)')
# parser.add_argument('--subtyping', action='store_true', default=False,
#                      help='subtyping problem')
# parser.add_argument('--bag_weight', type=float, default=0.7,
#                     help='clam: weight coefficient for bag-level loss (default: 0.7)')
# parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
# ### DFTD specific options
# parser.add_argument('--numLayer_Res', default=0, type=int)
# parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
# parser.add_argument('--epoch_step', default='[100]', type=str)
# parser.add_argument('--numGroup', default=4, type=int)
# parser.add_argument('--total_instance', default=4, type=int)
# parser.add_argument('--grad_clipping', default=5, type=float)
# parser.add_argument('--num_MeanInference', default=1, type=int)
# parser.add_argument('--distill_type', default='AFS', type=str)   ## MaxMinS, MaxS, AFS
# ### FRMIL specific options
# parser.add_argument('--shift_feature', action='store_true', default=False, help='shift feature')
# parser.add_argument('--drop_data', action='store_true', default=False, help='drop data')
# parser.add_argument('--balanced_sample', action='store_true', default=False, help='balanced bag')
# parser.add_argument('--n_heads', type=int, default=1, help='number of heads')
# parser.add_argument('--mag', type=float, default=1.0, help='magnitude')
# ### DFP
# parser.add_argument('--dfp', action='store_true', default=False)
# parser.add_argument('--dfp_discrim', action='store_true', default=False)
# parser.add_argument('--prompt_initialisation', type=str, default='gaussian', help='prompt init')
# parser.add_argument('--prompt_aggregation', type=str, default='multiply', choices=['multiply', 'add', 'prepend'], help='prompt aggregation method')
# parser.add_argument('--number_prompts', type=int, default=1)
# parser.add_argument('--prompt_epoch', type=int, default=10)
# ### HIPT
# parser.add_argument('--pretrain_4k',    type=str, default='None', help='Whether to initialize the 4K Transformer in HIPT', choices=['None', 'vit4k_xs_dino'])
# parser.add_argument('--pretrain_WSI',    type=str, default='None')
# parser.add_argument('--freeze_4k',      action='store_true', default=False, help='Whether to freeze the 4K Transformer in HIPT')
# parser.add_argument('--freeze_WSI',     action='store_true', default=False, help='Whether to freeze the WSI Transformer in HIPT')
# args = parser.parse_args()
# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logging.basicConfig(level=logging.INFO,
#                     filemode="w",
#                     format="%(name)s: %(asctime)s | %(filename)s:%(lineno)s |  %(message)s",
#                     filename=f"{args.fed_method}_{args.mil_method}_{args.ft_model}_{args.exp_code}_logs.txt")
# logger = logging.getLogger(__name__)
#
# args.results_dir = os.path.join(args.results_dir, f"{args.fed_method}_{args.mil_method}_{args.ft_model}_{args.exp_code}")
# if not os.path.exists(args.results_dir):
#     os.makedirs(args.results_dir)
# logger.info('Results will be saved in: {}'.format(args.results_dir))
# def seed_torch(seed=7):
#     import random
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if device.type == 'cuda':
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#
# if __name__ == '__main__':
#     overall_avg_acc, overall_acc_per_agent = [], {}
#     logger.info(f'Performing experiments: {args.exp_code} {args.fed_method} {args.mil_method} {args.ft_model}')
#     for rep in range(args.repeat):
#         runner = Meta(args, logger=logger)
#         logger.info(f'======================== Run {rep} Starts========================')
#         seed = int(datetime.now().timestamp())
#         seed_torch(seed)
#         if args.fed_method in ['fed_avg', 'fed_prox']:
#             best_accuracy, best_accuracy_per_agent = runner.forward_fedavg(rep)
#         elif args.fed_method == 'fed_dyn':
#             best_accuracy, best_accuracy_per_agent = runner.forward_feddyn(rep)
#         elif args.fed_method == 'scaffold':
#             best_accuracy, best_accuracy_per_agent = runner.forward_fedscaf(rep)
#         elif args.fed_method == 'moon':
#             best_accuracy, best_accuracy_per_agent = runner.forward_fedmoon(rep)
#         else:
#             raise NotImplementedError
#         overall_avg_acc.append(best_accuracy)
#         if len(overall_acc_per_agent)==0:
#             for i in range(len(best_accuracy_per_agent)):
#                 overall_acc_per_agent[i] = [best_accuracy_per_agent[i]]
#         else:
#             for i in range(len(best_accuracy_per_agent)):
#                 overall_acc_per_agent[i].append(best_accuracy_per_agent[i])
#         logger.info(f'======================== Run {rep} Ends========================')
#     logger.info(f'Accuracies: avg: {np.mean(overall_avg_acc):.4f} std: {np.std(overall_avg_acc):.4f} best: {np.max(overall_avg_acc):.4f}')
#     logger.info('Accuracies per agent: ')
#     for ag_idx in overall_acc_per_agent:
#         ag_acc = overall_acc_per_agent[ag_idx]
#         logger.info(f'Agent {ag_idx}: avg: {np.mean(ag_acc):.4f} std: {np.std(ag_acc):.4f} best: {np.max(ag_acc):.4f}')
#     logger.info(f'Best run: {np.argmax(overall_avg_acc)}')
#
#
# check if two synthetic images are same
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
def show_input_output_in_a_row(orig_image, output_image, figsize=(10, 4)):
	""" this function will help you to save the input and output in a row

	Args:
		orig_image: input image
		output_image: output image
		save_path (str): your desired destination to save
		figsize (tuple): save_image size
	"""
	# Create a figure and set the size
	plt.figure(figsize=figsize)

	# Add the first image on the left
	plt.subplot(1, 2, 1)
	plt.imshow(orig_image, cmap='gray')
	plt.title('Input')
	plt.axis('off')  # Turn off axis labels

	# Add the second image on the right
	plt.subplot(1, 2, 2)
	plt.imshow(output_image, cmap='gray')
	plt.title('Output')
	plt.axis('off')

# Load blurred image
blurred_image = cv2.imread('/Users/congcong/Library/CloudStorage/OneDrive-Personal/PhD/tut/9517_24T3/Lab1/MyAssignment/5483096/lab1/Task3.jpg', cv2.IMREAD_COLOR)


def upsharp_masking(image, kernel_size=9, sigmaX=0.0, alpha=1.0, beta=-1.0, unsharp_beta=7.0, gamma=0.0):
	""" Unsharp masking helper function for task3

	Args:
		image (_type_): input image
		kernel_size (int): kernerl size for Gaussian blur operation. Defaults to (0, 0).
		sigmaX (int): the Gaussian kernel standard deviation in X direction
		alpha (float): weight of the oringla image. Defaults to 1.0.
		beta (float): weight of blured image. Defaults to -1.0.
		unsharp_beta (float): weight of mask. Defaults to 7.0.
		gamma (float): scalar added to each sum. Default to 0.0
	"""
	gaussian_blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX)
	mask = cv2.addWeighted(image, alpha, gaussian_blur, beta, gamma)
	unsharpened_image = cv2.addWeighted(image, alpha, mask, unsharp_beta, gamma)
	return unsharpened_image

# deblured_image_unsharp = upsharp_masking(blurred_image)
# figsize=(10, 6)
# plt.figure(figsize=figsize)
#
# # Add the first image on the left
# plt.subplot(1, 2, 1)
# plt.imshow(blurred_image, cmap='gray')
# plt.title('Input')
# plt.axis('off')  # Turn off axis labels
#
# # Add the second image on the right
# plt.subplot(1, 2, 2)
# plt.imshow(deblured_image_unsharp, cmap='gray')
# plt.title('Output')
# plt.axis('off')
# plt.show()
# Convert to grayscale
# blurred_image = cv2.cvtColor(blurred_image, cv2.IMREAD_GRAYSCALE)
# print(blurred_image.shape)
# gray = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2GRAY)
TASK3_INPUT_IMAGE_PATH = '/Users/congcong/Library/CloudStorage/OneDrive-Personal/PhD/tut/9517_24T3/Lab1/MyAssignment/5483096/lab1/Task3.jpg'
# Read original image in grayscale
gray = cv2.imread(TASK3_INPUT_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
# Unsharp Masking
# Gaussian blur
# blurred = cv2.GaussianBlur(gray, (9, 9), 0.0)
# # Subtract blurred image from the original grayscale image
# unsharp_mask = cv2.addWeighted(blurred, 1, blurred, -1, 0.0)
unsharp_mask = upsharp_masking(gray)

# Laplacian Filter
# Laplacian operator
# laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)#, ksize=3, scale=0.65, delta=0)
# Convert to uint8 (8-bit single-channel image)
laplacian = cv2.convertScaleAbs(laplacian)
# Sharpen image by adding the Laplacian to the original image
laplacian_sharpened = cv2.addWeighted(gray, 1.0, laplacian, 5, 0)

# Display different filters
fig, axes = plt.subplots(3, 1, figsize=(30, 25))

# Original Image
axes[0].imshow(blurred_image, cmap='gray')
axes[0].set_title('Blurred Image (Input)')
axes[0].axis('off')

# Unsharp Masking result
axes[1].imshow(unsharp_mask, cmap='gray')
axes[1].set_title('Unsharp Masking')
axes[1].axis('off')

# Laplacian Sharpening result
axes[2].imshow(laplacian_sharpened, cmap='gray')
axes[2].set_title('Laplacian Sharpening')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# # Display results
# fig, axes = plt.subplots(1, 2, figsize=(15, 10))
#
# # Original Image
# axes[0].imshow(blurred_image)
# axes[0].set_title('Blurred Image (Input)')
# axes[0].axis('off')
#
# # Best-sharpened image
# best_sharpened = laplacian_sharpened
# axes[1].imshow(best_sharpened, cmap='gray')
# axes[1].set_title('Best Sharpened Image (Laplacian)')
# axes[1].axis('off')
#
# plt.tight_layout()
# plt.show()
