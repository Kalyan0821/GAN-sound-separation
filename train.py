import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
import time
import numpy as np
import os
from options.train_options import TrainOptions
from data.dataloader import create_dataloader
from models import components
from utils.train_utils import create_optimizer, softmax_normalization
from utils.utils import resample_logscale


def consistency_loss(predicted_masks, clip_ids, regression_loss, opt):
	""" predicted_masks: B x 1 x F x T
		clip_ids: B """

	predicted_mask_sums = dict()
	B, C, F, T = predicted_masks.shape

	for clip_id in clip_ids:
		predicted_mask_sums[clip_id.item()] = torch.zeros(C, F, T, device=opt.device)

	for i, clip_id in enumerate(clip_ids):
		predicted_mask_sums[clip_id.item()] += predicted_masks[i]  # C x F x T

	loss = torch.tensor(0., requires_grad=True)

	if opt.mask_loss_type == "BCE":
		for clip_id in clip_ids:
			predicted_mask_sums[clip_id.item()] = torch.clamp(predicted_mask_sums[clip_id.item()], min=0, max=1)
			loss = loss + regression_loss(predicted_mask_sums[clip_id.item()], torch.ones(C, F, T, device=opt.device))
	else:
		for clip_id in clip_ids:
			loss = loss + regression_loss(predicted_mask_sums[clip_id.item()], torch.ones(C, F, T, device=opt.device))

	loss /= B
	return loss


def disc_step(audio_mags, visuals, clip_ids, solo_audio_mags, solo_visuals,
			  net_visual, gen_unet, disc_encoder, disc_classifier,
			  gen_optimizer, disc_optimizer,
			  classification_loss, opt):

	# Take logs
	log_audio_mags = torch.log(audio_mags + 1e-10)
	log_solo_audio_mags = torch.log(solo_audio_mags + 1e-10)

	# Extract visual representations
	visual_features = net_visual(visuals)
	solo_visual_features = net_visual(solo_visuals)

	# Get separated audios
	if opt.softmax_constraint:
		raw_predicted_masks = gen_unet(log_audio_mags, visual_features)
		predicted_masks = softmax_normalization(raw_predicted_masks, clip_ids)
	else:
		predicted_masks = gen_unet(log_audio_mags, visual_features)

	separated_audio_mags = predicted_masks * audio_mags
	log_separated_audio_mags = torch.log(separated_audio_mags + 1e-10)

	# Generate real and fake predictions
	real_disc_encodings = disc_encoder(log_solo_audio_mags, solo_visual_features)  # use log_solo_audio_mags.detach() ??
	real_preds = disc_classifier(real_disc_encodings)

	fake_disc_encodings = disc_encoder(log_separated_audio_mags, visual_features)  # use log_separated_audio_mags.detach() ??
	fake_preds = disc_classifier(fake_disc_encodings)

	# Generate real and fake targets
	real_targets = torch.ones(real_preds.shape[0], 1, device=opt.device)
	fake_targets = torch.zeros(fake_preds.shape[0], 1, device=opt.device)

	# Loss
	disc_loss = classification_loss(real_preds, real_targets) + classification_loss(fake_preds, fake_targets)
	disc_loss.backward()
	disc_optimizer.step()

	# Get confidence scores
	with torch.no_grad():
		sigmoid = nn.Sigmoid()
		real_conf = torch.sum(sigmoid(real_preds))/real_preds.shape[0]  # apply sigmoid since model outputs logits, not probabilities
		fake_conf = 1 - torch.sum(sigmoid(fake_preds))/fake_preds.shape[0] 		

	return disc_loss.item(), real_conf.item(), fake_conf.item()


def gen_step(audio_mags, visuals, clip_ids,
			 net_visual, gen_unet, disc_encoder, disc_classifier,
			 gen_optimizer, disc_optimizer,
			 classification_loss, regression_loss,
			 opt):

	# Take logs
	log_audio_mags = torch.log(audio_mags + 1e-10)

	# Extract visual representations
	visual_features = net_visual(visuals)

	# Get separated audios
	if opt.softmax_constraint:
		raw_predicted_masks = gen_unet(log_audio_mags, visual_features)
		predicted_masks = softmax_normalization(raw_predicted_masks, clip_ids)
	else:
		predicted_masks = gen_unet(log_audio_mags, visual_features)


	separated_audio_mags = predicted_masks * audio_mags
	log_separated_audio_mags = torch.log(separated_audio_mags + 1e-10)

	# Generate fake predictions
	fake_disc_encodings = disc_encoder(log_separated_audio_mags, visual_features)  # use log_separated_audio_mags.detach() ??
	fake_preds = disc_classifier(fake_disc_encodings)

	# Generate fooling fake targets
	fake_targets = torch.ones(fake_preds.shape[0], 1, device=opt.device)

	# Loss components
	gen_loss_classification = classification_loss(fake_preds, fake_targets)
	gen_loss_consistency = consistency_loss(predicted_masks, clip_ids, regression_loss, opt)  # should be zero with SoftMax constraint
	
	# Total loss
	if opt.softmax_constraint:
		gen_loss = gen_loss_classification  # no need to include consistency loss due to SoftMax constraint
	else:
		gen_loss = gen_loss_classification + gen_loss_consistency*opt.consistency_loss_weight


	gen_loss.backward()
	gen_optimizer.step()

	return gen_loss_classification.item(), gen_loss_consistency.item()


########################################################################################

# Parse arguments
opt = TrainOptions().parse()
opt.device = torch.device("cuda")

# Dataloaders
dataloader_train = create_dataloader(opt)
print(f"Train batches: {len(dataloader_train)}\n")

# if opt.validation_on:
# 	opt.mode = "val"  # set temporalily
# 	dataloader_val = create_dataloader(opt)
# 	print(f"Validation batches: {len(dataloader_val)}")
# 	opt.mode = "train"

# Tensorboard
if opt.tensorboard:
	writer = SummaryWriter(logdir=f"./runs/{opt.experiment_id}")


# Initialize component networks
net_visual = components.build_visual(pool_type=opt.visual_pool,
									 fc_out=512,
									 weights=opt.weights_visual)

gen_unet = components.build_unet(unet_num_layers=opt.unet_num_layers,
								 ngf=opt.unet_ngf,
								 input_channels=opt.unet_input_nc,
								 output_channels=opt.unet_output_nc,
								 with_decoder=True,
								 weights=opt.weights_unet,
								 no_sigmoid=opt.softmax_constraint) 

disc_encoder = components.build_unet(unet_num_layers=opt.unet_num_layers,
									 ngf=opt.unet_ngf,
									 input_channels=opt.unet_input_nc,
									 output_channels=opt.unet_output_nc,
									 with_decoder=False)

disc_classifier = components.build_classifier(input_channels=1024, pool_type=opt.classifier_pool)

# Put components on GPU
net_visual.to(opt.device)
gen_unet.to(opt.device)
disc_encoder.to(opt.device)
disc_classifier.to(opt.device)

nets = (net_visual, gen_unet, disc_encoder, disc_classifier)

# Create optimizers
gen_optimizer = create_optimizer(nets=nets, mode="gen", opt=opt)
disc_optimizer = create_optimizer(nets=nets, mode="disc", opt=opt)

# Loss functions
classification_loss = nn.BCEWithLogitsLoss()

if opt.softmax_constraint:
	regression_loss = None  # Impose SoftMax constraint to make masks sum to 1
else:
	if opt.mask_loss_type == "L1":
		regression_loss = nn.L1Loss()
	elif opt.mask_loss_type == "L2":
		regression_loss = nn.MSELoss()
	elif opt.mask_loss_type == "BCE":
		regression_loss = nn.BCELoss()


disc_losses = []
real_confs = []
fake_confs = []
gen_losses_consistency = []
gen_losses_classification = []


# Train
batch_number = 0
for epoch in range(opt.num_epochs):

	for i, batch_dict in tqdm(enumerate(dataloader_train), file=sys.stdout):		
		batch_number += 1

		labels = batch_dict["labels"].to(opt.device).squeeze(dim=1).long()  # B, convert to long-int
		clip_ids = batch_dict["vids"].to(opt.device).squeeze(dim=1)  # B
		audio_mags = batch_dict["audio_mags"].to(opt.device)  # B x 1 x F x T (F=512, T=256)
		solo_audio_mags = batch_dict["solo_audio_mags"].to(opt.device)  # B x 1 x F x T
		visuals = batch_dict["visuals"].to(opt.device)  # B x 3 x n x n (n=224)
		solo_visuals = batch_dict["solo_visuals"].to(opt.device)  # B x 3 x n x n

		# Resample into log scale
		if opt.logscale_freq:
			audio_mags = resample_logscale(audio_mags, opt, f=256)  # B x 1 x f x T (f=256, T=256)
			solo_audio_mags = resample_logscale(solo_audio_mags, opt, f=256)  # B x 1 x f x T

		######################## Perform parameter updates ########################

		# Clear all gradients computed by backward()
		disc_optimizer.zero_grad()
		gen_optimizer.zero_grad()


		if batch_number % (opt.num_disc_updates+1) != 0:
			# discriminator update (frequent in wgan)
			batch_disc_loss, batch_real_conf, batch_fake_conf = disc_step(audio_mags, visuals, clip_ids, solo_audio_mags, solo_visuals,
																		  net_visual, gen_unet, disc_encoder, disc_classifier,
																		  gen_optimizer, disc_optimizer,
																		  classification_loss, opt)
			disc_losses.append(batch_disc_loss)
			real_confs.append(batch_real_conf)
			fake_confs.append(batch_fake_conf)

		elif batch_number % (opt.num_disc_updates+1) == 0:
			# generator update (infrequent in wgan)
			batch_gen_loss_classification, batch_gen_loss_consistency = gen_step(audio_mags, visuals, clip_ids,
																  net_visual, gen_unet, disc_encoder, disc_classifier,
																  gen_optimizer, disc_optimizer,
																  classification_loss, regression_loss,
																  opt)
			gen_losses_classification.append(batch_gen_loss_classification)
			gen_losses_consistency.append(batch_gen_loss_consistency)


		if batch_number % opt.display_freq == 0:

			avg_disc_loss = np.mean(disc_losses)
			avg_real_conf = np.mean(real_confs)
			avg_fake_conf = np.mean(fake_confs)
			avg_gen_loss_classification = np.mean(gen_losses_classification)
			avg_gen_loss_consistency = np.mean(gen_losses_consistency)

			if opt.tensorboard:
				# Log
				writer.add_scalar(tag="train_losses/disc_loss", scalar_value=avg_disc_loss, global_step=batch_number)
				writer.add_scalar(tag="train_losses/disc_real_conf", scalar_value=avg_real_conf, global_step=batch_number)
				writer.add_scalar(tag="train_losses/disc_fake_conf", scalar_value=avg_fake_conf, global_step=batch_number)

				writer.add_scalar(tag="train_losses/gen_loss_classification", scalar_value=avg_gen_loss_classification, global_step=batch_number)
				writer.add_scalar(tag="train_losses/gen_loss_consistency", scalar_value=avg_gen_loss_consistency, global_step=batch_number)

			# Print
			print(f"\nTraining progress @ Epoch: {epoch+1}, Iteration: {batch_number}\n")
			print(f"Generator classification loss: {avg_gen_loss_classification}")
			print(f"Generator consistency loss: {avg_gen_loss_consistency}")
			print(f"Discriminator loss: {avg_disc_loss}")
			print(f"Discriminator real confidence: {avg_real_conf}")
			print(f"Discriminator fake confidence: {avg_fake_conf}\n\n")
			# Reset
			disc_losses = []
			real_confs = []
			fake_confs = []
			gen_losses_classification = []
			gen_losses_consistency = []

		# if (opt.validation_on) and (batch_number % opt.validation_freq == 0):
		# 	# LOG
		# 	# PRINT
		# 	# SAVE best val model yet

	print(f"Saving latest model @ Epoch: {epoch+1}, Iteration: {batch_number}\n")
	torch.save(net_visual.state_dict(), os.path.join(opt.checkpoints_dir, opt.experiment_id, f"visual_epoch{epoch+1}.pth"))
	torch.save(gen_unet.state_dict(), os.path.join(opt.checkpoints_dir, opt.experiment_id, f"gen_unet_epoch{epoch+1}.pth"))
	torch.save(disc_encoder.state_dict(), os.path.join(opt.checkpoints_dir, opt.experiment_id, f"disc_encoder_epoch{epoch+1}.pth"))
	torch.save(disc_classifier.state_dict(), os.path.join(opt.checkpoints_dir, opt.experiment_id, f"disc_classifier_epoch{epoch+1}.pth"))

































