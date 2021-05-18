import torch

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



