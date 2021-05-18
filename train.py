import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import numpy as np
from options.train_options import TrainOptions
from data.dataloader import create_dataloader
from models import components
from utils.train_utils import create_optimizer
from utils.utils import resample_logscale


def consistency_loss(predicted_masks, clip_ids, regression_loss):
	""" predicted_masks: B x 1 x F x T
		clip_ids: B """

	predicted_mask_sums = dict()

	B, C, F, T = predicted_masks.shape

	for clip_id in clip_ids:
		predicted_mask_sums[clip_id] = torch.zeros(C, F, T)
	for i, clip_id in enumerate(clip_ids):
		predicted_mask_sums[clip_id] += predicted_masks[i]  # C x F x T

	loss = torch.tensor(0., requires_grad=True)
	for clip_id in clip_ids:
		loss += regression_loss(predicted_mask_sums[clip_id], torch.ones(C, F, T))
	loss /= B

	return loss


def disc_step(audio_mags, visuals, solo_audio_mags, solo_visuals,
			  net_visual, gen_unet, disc_encoder, disc_classifier,
			  gen_optimizer, disc_optimizer,
			  classification_loss):

	# Take logs
	log_audio_mags = torch.log(audio_mags + 1e-10)
	log_solo_audio_mags = torch.log(solo_audio_mags + 1e-10)

	# Extract visual representations
	visual_features = net_visual(visuals)
	solo_visual_features = net_visual(solo_visuals)

	# Get separated audios
	predicted_masks = gen_unet(log_audio_mags, visual_features)
	separated_audio_mags = predicted_masks * audio_mags
	log_separated_audio_mags = torch.log(separated_audio_mags + 1e-10)

	# Generate real and fake predictions
	real_disc_encodings = disc_encoder(log_solo_audio_mags, solo_visual_features)  # use log_solo_audio_mags.detach() ??
	real_preds = disc_classifier(real_disc_encodings)

	fake_disc_encodings = disc_encoder(log_separated_audio_mags, visual_features)  # use log_separated_audio_mags.detach() ??
	fake_preds = disc_classifier(fake_disc_encodings)

	# Generate real and fake targets
	real_targets = torch.ones(real_preds.shape[0], 1)
	fake_targets = torch.zeros(fake_preds.shape[0], 1)

	# Loss
	disc_loss = classification_loss(real_preds, real_targets) + classification_loss(fake_preds, fake_targets)
	disc_loss.backward()
	disc_optimizer.step()

	with torch.no_grad():
		real_acc = torch.sum(torch.round(real_preds)==1)/real_preds.shape[0]
        fake_acc = torch.sum(torch.round(fake_preds)==0)/fake_preds.shape[0] 


	return disc_loss.item(), real_acc.item(), fake_acc.item()


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
	predicted_masks = gen_unet(log_audio_mags, visual_features)
	separated_audio_mags = predicted_masks * audio_mags
	log_separated_audio_mags = torch.log(separated_audio_mags + 1e-10)

	# Generate fake predictions
	fake_disc_encodings = disc_encoder(log_separated_audio_mags, visual_features)  # use log_separated_audio_mags.detach() ??
	fake_preds = disc_classifier(fake_disc_encodings)

	# Generate fooling fake targets
	fake_targets = torch.ones(fake_preds.shape[0], 1)

	# Loss components
	gen_loss_classification = classification_loss(fake_preds, fake_targets)
	gen_loss_consistency = consistency_loss(predicted_masks, clip_ids, regression_loss)

	# Total loss
	gen_loss = gen_loss_classification + gen_loss_consistency*opt.consistency_loss_weight
	gen_loss.backward()
	gen_optimizer.step()

	return gen_loss.item(), gen_loss_consistency.item()

########################################################################################

# Parse arguments
opt = TrainOptions().parse()
opt.device = torch.device("cuda")

# Dataloaders
dataloader_train = create_dataloader(opt)
print(f"Train batches: {len(dataloader_train)}")

if opt.validation_on:
	opt.mode = "val"  # set temporalily
	dataloader_val = create_dataloader(opt)
	print(f"Validation batches: {len(dataloader_val)}")
	opt.mode = "train"

# Tensorboard
if opt.tensorboard:
	writer = SummaryWriter()
else:
	writer = None

# Initialize component networks
net_visual = components.build_visual(pool_type=opt.visual_pool,
									 fc_out=512,
									 weights=opt.weights_visual)

gen_unet = components.build_unet(unet_num_layers=opt.unet_num_layers,
								 ngf=opt.unet_ngf,
								 input_channels=opt.unet_input_nc,
						    	 output_channels=opt.unet_output_nc,
							     with_decoder=True,
								 weights=opt.weights_unet)

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
if opt.mask_loss_type == "L1":
	regression_loss = nn.L1Loss(reduction="sum")
if opt.mask_loss_type == "L2":
	regression_loss = nn.MSELoss(reduction="sum")
if opt.mask_loss_type == "BCE":
	regression_loss = nn.BCELoss(reduction="sum")

gen_losses = []
disc_losses = []
real_accs = []
fake_accs = []
gen_losses_consistency = []

# Train
batch_number = 0
for epoch in range(opt.num_epochs):

	for i, batch_dict in tqdm(enumerate(dataloader_train)):		
		batch_number += 1

		labels = batch_dict["labels"].squeeze(dim=1).long()  # B, convert to long-int
		clip_ids = batch_dict["vids"].squeeze(dim=1)  # B
		audio_mags = batch_dict["audio_mags"]  # B x 1 x F x T (F=512, T=256)
		solo_audio_mags = batch_dict["solo_audio_mags"]  # B x 1 x F x T
		visuals = batch_dict["visuals"]  # B x 3 x n x n (n=224)
		solo_visuals = batch_dict["solo_visuals"]  # B x 3 x n x n

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
			batch_disc_loss, batch_real_acc, batch_fake_acc = disc_step(audio_mags, visuals, solo_audio_mags, solo_visuals,
					  													net_visual, gen_unet, disc_encoder, disc_classifier,
																		gen_optimizer, disc_optimizer,
																		classification_loss)
			disc_losses.append(batch_disc_loss)
			real_accs.append(batch_real_acc)
			fake_accs.append(batch_fake_acc)

		elif batch_number % (opt.num_disc_updates+1) == 0:
			# generator update (infrequent in wgan)
			batch_gen_loss, batch_gen_loss_consistency = gen_step(audio_mags, visuals, clip_ids
					 											  net_visual, gen_unet, disc_encoder, disc_classifier,
																  gen_optimizer, disc_optimizer,
																  classification_loss, regression_loss,
																  opt)
			gen_losses.append(batch_gen_loss)
			gen_losses_consistency.append(batch_gen_loss_consistency)

		if batch_number % opt.display_freq == 0:
			avg_gen_loss = np.mean(gen_losses)
			avg_disc_loss = np.mean(disc_losses)
			avg_real_acc = np.mean(real_accs)
			avg_fake_acc = np.mean(fake_accs)
			avg_gen_loss_consistency = np.mean(gen_losses_consistency)
			if opt.tensorboard:
				# Log
				writer.add_scalar(tag="train_losses/gen_loss", scalar_value=avg_gen_loss, global_step=batch_number, display_name="gen_loss")
				writer.add_scalar(tag="train_losses/disc_loss", scalar_value=avg_disc_loss, global_step=batch_number, display_name="disc_loss")
				writer.add_scalar(tag="train_losses/disc_real_acc", scalar_value=avg_real_acc, global_step=batch_number, display_name="disc_real_acc")
				writer.add_scalar(tag="train_losses/disc_fake_acc", scalar_value=avg_fake_acc, global_step=batch_number, display_name="disc_fake_acc")
				writer.add_scalar(tag="train_losses/gen_loss_consistency", scalar_value=avg_gen_loss_consistency, global_step=batch_number, display_name="gen_loss_consistency")
			# Print
			print(f"\nTraining progress @ Epoch: {epoch}, Iteration: {batch_number}")
			print(f"Generator loss: {avg_gen_loss}")
			print(f"Discriminator loss: {avg_disc_loss}")
			print(f"Discriminator real accuracy: {avg_real_acc}")
			print(f"Discriminator fake accuracy: {avg_fake_acc}")
			print(f"Generator consistency loss: {avg_gen_loss_consistency}")
			# Reset
			disc_losses = []
			real_accs = []
			fake_accs = []
			gen_losses = []
			gen_losses_consistency = []

		# if (opt.validation_on) and (batch_number % opt.validation_freq == 0):
		# 	# LOG
		# 	# PRINT
		# 	# SAVE best val model yet

	print(f"Saving latest model @ Epoch: {epoch}, Iteration: {batch_number}\n")
	torch.save(net_visual.state_dict(), os.path.join(opt.checkpoints_dir, opt.experiment_id, f"visual_epoch{epoch}.pth"))
	torch.save(gen_unet.state_dict(), os.path.join(opt.checkpoints_dir, opt.experiment_id, f"unet_epoch{epoch}.pth"))
	torch.save(disc_encoder.state_dict(), os.path.join(opt.checkpoints_dir, opt.experiment_id, f"disc_encoder_epoch{epoch}.pth"))
	torch.save(disc_classifier.state_dict(), os.path.join(opt.checkpoints_dir, opt.experiment_id, f"disc_classifier_epoch{epoch}.pth"))

































