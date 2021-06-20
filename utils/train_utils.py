import torch
import torch.nn as nn


def create_optimizer(nets, mode, opt):
	net_visual, gen_unet, disc_encoder, disc_classifier = nets
	if mode == "gen":
		param_groups = [{"params": net_visual.parameters(), "lr": opt.lr_visual},
						{"params": gen_unet.parameters(), "lr": opt.lr_unet}]
	elif mode == "disc":
		param_groups = [{"params": net_visual.parameters(), "lr": opt.lr_visual},
						{"params": disc_encoder.parameters(), "lr": opt.lr_unet},
						{"params": disc_classifier.parameters(), "lr": opt.lr_classifier}]

	if opt.optimizer == "sgd":
		return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
	elif opt.optimizer == 'adam':
		return torch.optim.Adam(param_groups, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)



def softmax_normalization(raw_predicted_masks, clip_ids):
	""" raw_predicted_masks: B x 1 x F x T
		clip_ids: B """

	raw_predicted_mask_aggregates = dict()
	normalized_predicted_mask_aggregates = dict()
	predicted_masks_list = []

	B, C, F, T = raw_predicted_masks.shape

	for clip_id in clip_ids:
		raw_predicted_mask_aggregates[clip_id.item()] = []
		normalized_predicted_mask_aggregates[clip_id.item()] = [] 

	for i, clip_id in enumerate(clip_ids):
		raw_predicted_mask_aggregates[clip_id.item()].append(raw_predicted_masks[i:(i+1)])  # [list of 1xCxFxT tensors]

	for i, clip_id in enumerate(clip_ids):
		normalized_predicted_mask_aggregates[clip_id.item()] = nn.Softmax(
															   torch.stack(raw_predicted_mask_aggregates[clip_id.item()], dim=0),
															   dim=0)  # k x C x F x T
		predicted_masks_list.append(normalized_predicted_mask_aggregates)

	predicted_masks = torch.stack(predicted_masks_list, dim=0)
	assert predicted_masks.shape == raw_predicted_masks.shape

	return predicted_masks  # B x 1 x F x T

