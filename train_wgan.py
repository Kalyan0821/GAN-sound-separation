#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from options.train_options import TrainOptions
from data.dataloader import create_dataloader
from data.dataset_MUSIC import MUSICDataset
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from scipy.misc import imsave   
import scipy.io.wavfile as wavfile
import numpy as np
import torch
from torch.autograd import Variable
import librosa
from utils import utils,viz
from utils.utils import warpgrid
from models import criterion
import torch.nn.functional as F
from tqdm import tqdm

def create_optimizer(nets, opt, mode):
    (net_visual, net_unet, net_crit) = nets
    if mode == 'gen':
        param_groups = [{'params': net_visual.parameters(), 'lr': opt.lr_visual},
                    {'params': net_unet.parameters(), 'lr': opt.lr_unet}]
    else:
        param_groups = [{'params': net_visual.parameters(), 'lr': opt.lr_visual},
                    {'params': net_unet.parameters(), 'lr': opt.lr_unet},
                    {'params': net_crit.parameters(), 'lr': opt.lr_classifier}]
    if opt.optimizer == 'sgd':
        return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def decrease_learning_rate(optimizer, decay_factor=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

def save_visualization(vis_rows, outputs, batch_data, save_dir, opt):
    # fetch data and predictions
    mag_mix = batch_data['audio_mix_mags']
    phase_mix = batch_data['audio_mix_phases']
    visuals = batch_data['visuals']

    pred_masks_ = outputs['pred_mask']
    gt_masks_ = outputs['gt_mask']
    mag_mix_ = outputs['audio_mix_mags']
    weight_ = outputs['weight']
    visual_object = outputs['visual_object']
    gt_label = outputs['gt_label']
    _, pred_label = torch.max(output['pred_label'], 1)
    label_list = ['Banjo', 'Cello', 'Drum', 'Guitar', 'Harp', 'Harmonica', 'Oboe', 'Piano', 'Saxophone', \
                    'Trombone', 'Trumpet', 'Violin', 'Flute','Accordion', 'Horn']

    # unwarp log scale
    B = mag_mix.size(0)
    if opt.log_freq:
        grid_unwarp = torch.from_numpy(utils.warpgrid(B, opt.stft_frame//2+1, gt_masks_.size(3), warp=False)).to(opt.device)
        pred_masks_linear = F.grid_sample(pred_masks_, grid_unwarp)
        gt_masks_linear = F.grid_sample(gt_masks_, grid_unwarp)
    else:
        pred_masks_linear = pred_masks_
        gt_masks_linear = gt_masks_

    # convert into numpy
    mag_mix = mag_mix.numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.numpy()
    weight_ = weight_.detach().cpu().numpy()
    pred_masks_ = pred_masks_.detach().cpu().numpy()
    pred_masks_linear = pred_masks_linear.detach().cpu().numpy()
    gt_masks_ = gt_masks_.detach().cpu().numpy()
    gt_masks_linear = gt_masks_linear.detach().cpu().numpy()
    visual_object = visual_object.detach().cpu().numpy()
    gt_label = gt_label.detach().cpu().numpy()
    pred_label = pred_label.detach().cpu().numpy()

    # loop over each example
    for j in range(min(B, opt.num_visualization_examples)):
        row_elements = []

        # video names
        prefix = str(j) + '-' + label_list[int(gt_label[j])] + '-' + label_list[int(pred_label[j])]
        utils.mkdirs(os.path.join(save_dir, prefix))

        # save mixture
        mix_wav = utils.istft_coseparation(mag_mix[j, 0], phase_mix[j, 0], hop_length=opt.stft_hop)
        mix_amp = utils.magnitude2heatmap(mag_mix_[j, 0])
        weight = utils.magnitude2heatmap(weight_[j, 0], log=False, scale=100.)
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        filename_weight = os.path.join(prefix, 'weight.jpg')
        imsave(os.path.join(save_dir, filename_mixmag), mix_amp[::-1, :, :])
        imsave(os.path.join(save_dir, filename_weight), weight[::-1, :])
        wavfile.write(os.path.join(save_dir, filename_mixwav), opt.audio_sampling_rate, mix_wav)
        row_elements += [{'text': prefix}, {'image': filename_mixmag, 'audio': filename_mixwav}]

        # GT and predicted audio reconstruction
        gt_mag = mag_mix[j, 0] * gt_masks_linear[j, 0]
        gt_wav = utils.istft_coseparation(gt_mag, phase_mix[j, 0], hop_length=opt.stft_hop)
        pred_mag = mag_mix[j, 0] * pred_masks_linear[j, 0]
        preds_wav = utils.istft_coseparation(pred_mag, phase_mix[j, 0], hop_length=opt.stft_hop)

        # output masks
        filename_gtmask = os.path.join(prefix, 'gtmask.jpg')
        filename_predmask = os.path.join(prefix, 'predmask.jpg')
        gt_mask = (np.clip(gt_masks_[j, 0], 0, 1) * 255).astype(np.uint8)
        pred_mask = (np.clip(pred_masks_[j, 0], 0, 1) * 255).astype(np.uint8)
        imsave(os.path.join(save_dir, filename_gtmask), gt_mask[::-1, :])
        imsave(os.path.join(save_dir, filename_predmask), pred_mask[::-1, :])

        # ouput spectrogram (log of magnitude, show colormap)
        filename_gtmag = os.path.join(prefix, 'gtamp.jpg')
        filename_predmag = os.path.join(prefix, 'predamp.jpg')
        gt_mag = utils.magnitude2heatmap(gt_mag)
        pred_mag = utils.magnitude2heatmap(pred_mag)
        imsave(os.path.join(save_dir, filename_gtmag), gt_mag[::-1, :, :])
        imsave(os.path.join(save_dir, filename_predmag), pred_mag[::-1, :, :])

        # output audio
        filename_gtwav = os.path.join(prefix, 'gt.wav')
        filename_predwav = os.path.join(prefix, 'pred.wav')
        wavfile.write(os.path.join(save_dir, filename_gtwav), opt.audio_sampling_rate, gt_wav)
        wavfile.write(os.path.join(save_dir, filename_predwav), opt.audio_sampling_rate, preds_wav)

        row_elements += [
                {'image': filename_predmag, 'audio': filename_predwav},
                {'image': filename_gtmag, 'audio': filename_gtwav},
                {'image': filename_predmask},
                {'image': filename_gtmask}]

        row_elements += [{'image': filename_weight}]
        vis_rows.append(row_elements)

#used to display validation loss
def display_val(model, crit, writer, index, dataset_val, opt):
    # remove previous viz results
    save_dir = os.path.join('.', opt.checkpoints_dir, opt.name, 'visualization')
    utils.mkdirs(save_dir)

    #initial results lists
    accuracies = []
    classifier_losses = []
    coseparation_losses = []

    # initialize HTML header
    visualizer = viz.HTMLVisualizer(os.path.join(save_dir, 'index.html'))
    header = ['Filename', 'Input Mixed Audio']
    header += ['Predicted Audio' 'GroundTruth Audio', 'Predicted Mask','GroundTruth Mask', 'Loss weighting']
    visualizer.add_header(header)
    vis_rows = []

    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            if i < opt.validation_batches:
                output = model.forward(val_data)
                loss_classification = crit['loss_classification']
                classifier_loss = loss_classification(output['pred_label'], Variable(output['gt_label'], requires_grad=False)) * opt.classifier_loss_weight
                coseparation_loss = get_coseparation_loss(output, opt, crit['loss_coseparation'])
                classifier_losses.append(classifier_loss.item()) 
                coseparation_losses.append(coseparation_loss.item())
                gt_label = output['gt_label']
                _, pred_label = torch.max(output['pred_label'], 1)
                accuracy = torch.sum(gt_label == pred_label).item() * 1.0 / pred_label.shape[0]
                accuracies.append(accuracy)
            else:
                if opt.validation_visualization:
                    output = model.forward(val_data)
                    save_visualization(vis_rows, output, val_data, save_dir, opt) #visualize one batch
                break

    avg_accuracy = sum(accuracies)/len(accuracies)
    avg_classifier_loss = sum(classifier_losses)/len(classifier_losses)
    avg_coseparation_loss = sum(coseparation_losses)/len(coseparation_losses)
    if opt.tensorboard:
        writer.add_scalar('data/val_classifier_loss', avg_classifier_loss, index)
        writer.add_scalar('data/val_accuracy', avg_accuracy, index)
        writer.add_scalar('data/val_coseparation_loss', avg_coseparation_loss, index)
    print('val accuracy: %.3f' % avg_accuracy)
    print('val classifier loss: %.3f' % avg_classifier_loss)
    print('val coseparation loss: %.3f' % avg_coseparation_loss)
    return avg_coseparation_loss + avg_classifier_loss

#get_consistency_loss(pred_solo_spec,fake_audio_mix_mags, fake_vid)
def get_consistency_loss(pred_spec, GT_mix_spec, fake_vid, L2_loss_criterion, opt):

    unique_vid = torch.unique(fake_vid)

    B,C,F,T = pred_spec.shape
    new_pred = torch.zeros(len(unique_vid),C,F,T).to(opt.device)
    gt = torch.zeros(len(unique_vid),C,F,T).to(opt.device)

    for i,unq_vid_id in enumerate(unique_vid):
        for j,vid_id in enumerate(fake_vid):
            if unq_vid_id == vid_id:
                #print([i,j])
                new_pred[i] += pred_spec[j]
                gt[i] += GT_mix_spec[j]

    consistency_loss = L2_loss_criterion(new_pred, Variable(gt, requires_grad=False))

    return consistency_loss

#gradient = get_gradient(net_unet, net_visual, net_crit, fake_separated_spectrogram.detach(),fake_visuals.detach(), real_audio_mix_mags, real_visuals, epsilon)
def get_gradient(net_unet, net_visual, net_crit, real_spec, real_visual, fake_spec, fake_visual, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real spectrogram + visual features
        fake: a batch of fake spectrogram + visual features
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    B = np.min((real_spec.shape[0], fake_spec.shape[0]))

    mixed_spec = real_spec[0:B] * epsilon[0:B] + fake_spec[0:B] * (1 - epsilon[0:B])
    mixed_visual = real_visual[0:B] * epsilon[0:B] + fake_visual[0:B] * (1 - epsilon[0:B])
    
    mixed_spec_feature = net_unet.forward_encoder(mixed_spec)
    mixed_visual_feature = net_visual(mixed_visual)

    # Calculate the critic's scores on the mixed images
    mixed_scores = net_crit(mixed_spec_feature, mixed_visual_feature)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        #### START CODE HERE ####
        inputs=(mixed_spec,mixed_visual),
        outputs=mixed_scores,
        #### END CODE HERE ####
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )

    return gradient

def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient0 = gradient[0].view(len(gradient[0]), -1)
    gradient1 = gradient[1].view(len(gradient[1]), -1)

    gradient = torch.cat([gradient0, gradient1], dim=1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1
    #### START CODE HERE ####
    penalty = torch.mean((gradient_norm - 1)**2)
    #### END CODE HERE ####
    return penalty

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty 
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    #### START CODE HERE ####
    crit_loss = (crit_fake_pred.mean() - crit_real_pred.mean()) + c_lambda * gp
    #### END CODE HERE ####
    return crit_loss


def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    #### START CODE HERE ####
    gen_loss = -crit_fake_pred.mean()
    #### END CODE HERE ####
    return gen_loss

def crit_forward_backward_pass(net_visual, net_unet, net_crit, fake_visuals, fake_audio_mix_mags, real_visuals, real_audio_mix_mags, opt):
    # step 2 : forward pass - generator

    # pass through visual stream and extract visual features
    fake_visual_feature = net_visual(Variable(fake_visuals, requires_grad=False))
    real_visual_feature = net_visual(Variable(real_visuals, requires_grad=False))

    # audio-visual feature fusion through UNet and predict mask
    fake_audio_log_mags = torch.log(fake_audio_mix_mags).detach()
    fake_mask_prediction = net_unet(fake_audio_log_mags, fake_visual_feature)

    # masking the spectrogram of mixed audio to perform separation
    fake_separated_spectrogram = fake_audio_mix_mags * fake_mask_prediction
    
    # step 3 : forward pass - disc
    fake_spec_features = net_unet.forward_encoder(fake_separated_spectrogram.detach())
    real_spec_features = net_unet.forward_encoder(real_audio_mix_mags.detach())

    crit_fake_pred = net_crit(fake_spec_features, fake_visual_feature)
    crit_real_pred = net_crit(real_spec_features, real_visual_feature)

    # step 4 : backward pass - disc
    B = fake_audio_mix_mags.size(0)
    epsilon = torch.rand(B, 1, 1, 1, device=opt.device, requires_grad=True)
    gradient = get_gradient(net_unet, net_visual, net_crit, fake_separated_spectrogram.detach(),fake_visuals.detach(), real_audio_mix_mags, real_visuals, epsilon)
    gp = gradient_penalty(gradient)
    crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, opt.c_lambda)

    return crit_loss

def gen_forward_backward_pass(net_visual, net_unet, net_crit, fake_visuals, fake_audio_mix_mags):
    # pass through visual stream and extract visual features
    fake_visual_feature = net_visual(Variable(fake_visuals, requires_grad=False))
    
    # audio-visual feature fusion through UNet and predict mask
    fake_audio_log_mags = torch.log(fake_audio_mix_mags).detach()
    fake_mask_prediction = net_unet(fake_audio_log_mags, fake_visual_feature)

    # masking the spectrogram of mixed audio to perform separation
    fake_separated_spectrogram = fake_audio_mix_mags * fake_mask_prediction
    
    # step 3 : forward pass - disc
    fake_spec_features = net_unet.forward_encoder(fake_separated_spectrogram)
    crit_fake_pred = net_crit(fake_spec_features,fake_visual_feature)

    return crit_fake_pred, fake_separated_spectrogram

#parse arguments
opt = TrainOptions().parse()
opt.device = torch.device("cuda")

if opt.with_additional_scene_image:
    opt.number_of_classes = opt.number_of_classes + 1

############################
## construct data loader ###
############################
data_loader_real = create_dataloader(opt)
#dataset_real = data_loader_real.load_data()
dataset_real_size = len(data_loader_real)
print('#REAL training images = %d' % dataset_real_size)


if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(comment=opt.name)
else:
    writer = None

# Network Builders
builder = ModelBuilder()
net_visual = builder.build_visual(
        pool_type=opt.visual_pool,
        fc_out = 512,
        weights=opt.weights_visual)
net_visual.to(opt.device)
net_unet = builder.build_unet(
        unet_num_layers = opt.unet_num_layers,
        ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc,
        output_nc=opt.unet_output_nc,
        weights=opt.weights_unet)
net_unet.to(opt.device)
net_crit = builder.build_discriminator(
        ngf=opt.unet_ngf,
        weights=opt.weights_classifier)
net_crit.to(opt.device)
nets = (net_visual, net_unet, net_crit)


# Set up optimizer
gen_opt = create_optimizer(nets, opt, 'gen')
crit_opt = create_optimizer(nets, opt, 'crit')
L2_loss_criterion = torch.nn.MSELoss()

#initialization
total_batches = 0
cur_step = 0
generator_losses = []
critic_losses = []
mean_iteration_critic_loss = 0

for epoch in range(1 + opt.epoch_count, opt.niter+1):
        torch.cuda.synchronize()
        with torch.autograd.set_detect_anomaly(True):
            for real_data in tqdm(data_loader_real):
                    total_batches += 1

                    fake_audio_mix_mags =  real_data['audio_mags'].to(opt.device)
                    real_audio_solo_mags = real_data['solo_audio_mags'].to(opt.device)
                    fake_visuals = real_data['visuals'].to(opt.device)
                    real_visuals = real_data['solo_visuals'].to(opt.device)
                    real_labels = real_data['labels'].to(opt.device)
                    fake_vid = real_data['vids']

                    fake_audio_mix_mags = fake_audio_mix_mags + 1e-10
                    real_audio_solo_mags = real_audio_solo_mags + 1e-10
                    # warp the spectrogram
                    B_fake, B_real = fake_audio_mix_mags.size(0), real_audio_solo_mags.size(0)
                    T_fake, T_real = fake_audio_mix_mags.size(3), real_audio_solo_mags.size(3)
                     
                    if opt.log_freq:
                        grid_warp_real = torch.from_numpy(warpgrid(B_real, 256, T_real, warp=True)).to(opt.device)
                        real_audio_mags = F.grid_sample(real_audio_solo_mags, grid_warp_real)

                        grid_warp_fake = torch.from_numpy(warpgrid(B_fake, 256, T_fake, warp=True)).to(opt.device)
                        fake_audio_mags = F.grid_sample(fake_audio_mix_mags, grid_warp_fake)

                    crit_opt.zero_grad()
                    net_visual.zero_grad()
                    net_unet.zero_grad()
                    net_crit.zero_grad()

                    crit_loss = crit_forward_backward_pass(net_visual, net_unet, net_crit, fake_visuals, fake_audio_mags, real_visuals, real_audio_mags, opt)

                    # Keep track of the average critic loss in this batch
                    mean_iteration_critic_loss += crit_loss.item() / opt.crit_repeats
                    # Update gradients
                    crit_loss.backward()
                    # Update optimizer
                    crit_opt.step()

                    ### Update generator once every opt.crit_repeats iterations ###
                    if total_batches % opt.crit_repeats == 0:
                        gen_opt.zero_grad()
                        net_visual.zero_grad()
                        net_unet.zero_grad()
                        net_crit.zero_grad()
                        #print(fake_vid)

                        crit_fake_pred, pred_solo_spec = gen_forward_backward_pass(net_visual, net_unet, net_crit, fake_visuals, fake_audio_mags)
                        consistency_loss = get_consistency_loss(pred_solo_spec,fake_audio_mags, fake_vid, L2_loss_criterion, opt)
                        gen_loss = get_gen_loss(crit_fake_pred) 
                        total_gen_loss = gen_loss + 0.1 * consistency_loss
                        total_gen_loss.backward()

                        # Update the weights
                        gen_opt.step()

                        # Keep track of the average generator loss
                        #generator_losses += [gen_loss.item()]
                        print(f'{total_batches}\t{mean_iteration_critic_loss}\t{gen_loss.item()}\t{consistency_loss.item()}\t{total_gen_loss.item()}')
                        writer.add_scalar('data/train_crit', mean_iteration_critic_loss, total_batches)
                        writer.add_scalar('data/train_gen_c', gen_loss.item(), total_batches)
                        writer.add_scalar('data/train_consistency_loss', consistency_loss.item(), total_batches)
                        writer.add_scalar('data/train_gen_loss', total_gen_loss.item(), total_batches)
                        
                        mean_iteration_critic_loss = 0
                        net_visual.to('cpu')
                        net_unet.to('cpu')
                        net_crit.to('cpu')
                        net_visual.to(opt.device)
                        net_unet.to(opt.device)
                        net_crit.to(opt.device)

                    if total_batches%opt.save_latest_freq == 0:
                        print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_batches))
                        torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, f'latest_visual_latest.pth'))
                        torch.save(net_unet.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, f'latest_audio_latest.pth'))
                        torch.save(net_crit.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, f'latest_crit_latest.pth'))

        print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_batches))
        torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, f'epoch_{epoch}_steps_{total_batches}_visual_latest.pth'))
        torch.save(net_unet.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, f'epoch_{epoch}_steps_{total_batches}_audio_latest.pth'))
        torch.save(net_crit.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, f'epoch_{epoch}_steps_{total_batches}_crit_latest.pth'))
            
            