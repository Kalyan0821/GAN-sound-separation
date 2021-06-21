from PIL import Image
import numpy as np
import random
import os
import h5py
import librosa
import soundfile as sf
from scipy.io.wavfile import write as write_wav
import torch
from tqdm import tqdm
import sys
import time
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from data.dataset_utils import get_vid_path_MUSIC, get_audio_path_MUSIC, get_frames_path_MUSIC, get_ground_truth_labels_MUSIC
from data.dataset_utils import sample_object_detections, sample_audio, augment_audio, generate_spectrogram_magphase, augment_image
from data.dataloader import create_dataloader
from models import components
from options.train_options import TrainOptions
from utils import utils

opt = TrainOptions().parse()
opt.device = torch.device("cuda")
opt.hop_size = 0.05

def clip_audio(audio):
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def save_wav(filename, data, audio_rate):
    if data.ndim > 1 and data.shape[0] == 2:
        data = data.T

    write_wav(filename, audio_rate, data)

def get_separated_audio(mag_mix, phase_mix, pred_masks_, opt, log_freq=1):
	# unwarp log scale
	B = mag_mix.size(0)
	if log_freq:
		grid_unwarp = torch.from_numpy(utils.warpgrid(B, opt.stft_frame//2+1, pred_masks_.size(3), warp=False)).to(opt.device)
		pred_masks_linear = F.grid_sample(pred_masks_, grid_unwarp)
	else:
		pred_masks_linear = pred_masks_
	# convert into numpy
	mag_mix = mag_mix.numpy()
	phase_mix = phase_mix.numpy()
	pred_masks_linear = pred_masks_linear.detach().cpu().numpy()
	pred_mag = mag_mix[0, 0] * pred_masks_linear[0, 0]
	preds_wav = utils.istft_reconstruction(pred_mag, phase_mix[0, 0], hop_length=opt.stft_hop, length=opt.audio_window)
	return preds_wav


##################
## INIT models
##################
opt.weights_visual = './checkpoints/music_vanilla_less_consistency/visual_epoch7.pth'
opt.weights_unet = './checkpoints/music_vanilla_less_consistency/gen_unet_epoch7.pth'

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
# Put components on GPU
net_visual.to(opt.device)
gen_unet.to(opt.device)


detector_labels = ['__background__',
                    'Banjo', 'Cello', 'Drum', 'Guitar',
                    'Harp', 'Harmonica', 'Oboe', 'Piano',
                    'Saxophone', 'Trombone', 'Trumpet', 'Violin',
                    'Flute', 'Accordion', 'Horn']


detector_to_MUSIC_label = {
                    "Accordion": "accordion",
                    "Guitar": "acoustic_guitar",
                    "Cello": "cello",
                    "Oboe": "clarinet",
                    "Flute": "flute",
                    "Saxophone": "saxophone",
                    "Trumpet": "trumpet",
                    "Horn": "tuba",
                    "Violin": "violin",
                    "Piano": "xylophone"
                    }

################
## load data
##############
detection_dic = dict()  # {video_name: [clip_detection_npy_paths]} dict

#detections_file = os.path.join(opt.all_paths_dir, opt.dataset, "test.txt")
detections_file = os.path.join(opt.all_paths_dir, opt.dataset, "val.txt")

with open(detections_file, 'r') as f:
    detections = [s[:-1] for s in f.readlines()]  # list of all .npy clip detection file paths (strip the '\n')
    for detection in detections:  # iterate through all .npy paths
        vid_path = get_vid_path_MUSIC(detection)  # get name of video the clip belongs to
        if vid_path in detection_dic:
            detection_dic[vid_path].append(detection)
        else:
            detection_dic[vid_path] = [detection]

# vision transforms
vision_transform_list = [transforms.Resize((224,224)), transforms.ToTensor()]
if opt.subtract_mean:
    vision_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

vision_transform = transforms.Compose(vision_transform_list)

#NUM_VIDEOS = ##########??????????????/
keys = list(detection_dic.keys())
num_Videos = len(keys)

####################
# decode one video
###################

for vid_id in tqdm(range(num_Videos)):
	# get 1 random video to separate- "idx" plays no role here
	video = keys[vid_id]
	ground_truth_labels = get_ground_truth_labels_MUSIC(video)		

	# loop over all clips
	N_clips = len(detection_dic[video])				#?????????????
	clip_id = 1 ####### MODIFY THIS
	clip_det_path = detection_dic[video][clip_id]  
	#clip_det_path = random.choice(detection_dic[video])  # randomly sample 1 clip from the video
	clip_det_bbs = sample_object_detections(np.load(clip_det_path))  # B x 7 array (B discovered classes in the clip)
	detected_labels_idx = clip_det_bbs[:, 1].astype(int)  # all B detected class indices
	assert clip_det_bbs.shape[0] != 0  
	assert 0 not in clip_det_bbs[:, 1]  # Make sure __background__ is not a detected class


	# audio
	audio_path = get_audio_path_MUSIC(clip_det_path)
	audio, audio_rate = librosa.load(audio_path, sr=opt.audio_sampling_rate)  # load audio of clip at 11025 Hz (default)

	## sliding window decoding
	audio_length = len(audio)#opt.audio_window#len(audio)
	num_objects = len(clip_det_bbs)
	opt_folder = video[video.find('TOP_detection_results')+len('TOP_detection_results')+1:].replace('/','_')

	#print(num_objects)
	for i in range(num_objects):
		# get path of the single randomly sampled frame of the ith BB-image
		frame_path = os.path.join(get_frames_path_MUSIC(clip_det_path), str(int(clip_det_bbs[i, 0])).zfill(6)+".png")			
		# Crop out the BB image from the sampled frame for each discovered class in the clip
		object_image = Image.open(frame_path).convert('RGB').crop(
			(clip_det_bbs[i,-4], clip_det_bbs[i,-3], clip_det_bbs[i,-2], clip_det_bbs[i,-1]))

		# reshape and normalize each BB image
		objects_visuals = vision_transform(object_image).unsqueeze(0)

		label = clip_det_bbs[i, 1] - 1  # convert class label to zero-based index, i.e., [0, 15] => [-1, 14]
		objects_labels= (label)
		
		#perform separation over the whole audio using a sliding window approach
		overlap_count = np.zeros((audio_length))
		sep_audio = np.zeros((audio_length))
		sliding_window_start = 0
		data = {}
		samples_per_window = opt.audio_window
		while sliding_window_start + samples_per_window < audio_length:
			sliding_window_end = sliding_window_start + samples_per_window
			audio_segment = audio[sliding_window_start:sliding_window_end]
			audio_mix_mags, audio_mix_phases = generate_spectrogram_magphase(audio_segment, opt.stft_frame, opt.stft_hop) 
			audio_mix_mags = torch.FloatTensor(audio_mix_mags).unsqueeze(0)
			audio_mix_phases = torch.FloatTensor(audio_mix_phases).unsqueeze(0)
			audio_mags = audio_mix_mags.to(opt.device) #dont' care for testing

			#separate for video 1
			log_audio_mags = torch.log(audio_mags + 1e-10)
			visual_features = net_visual(objects_visuals.to(opt.device))

			predicted_masks = gen_unet(log_audio_mags, visual_features)
			
			reconstructed_signal = get_separated_audio(audio_mix_mags, audio_mix_phases, predicted_masks, opt, log_freq=1)
			
			sep_audio[sliding_window_start:sliding_window_end] = sep_audio[sliding_window_start:sliding_window_end] + reconstructed_signal
			overlap_count[sliding_window_start:sliding_window_end] = overlap_count[sliding_window_start:sliding_window_end] + 1
			sliding_window_start = sliding_window_start + int(opt.hop_size * opt.audio_sampling_rate)

		#deal with the last segment
		audio_segment = audio[-samples_per_window:]
		audio_mix_mags, audio_mix_phases = generate_spectrogram_magphase(audio_segment, opt.stft_frame, opt.stft_hop) 
		audio_mix_mags = torch.FloatTensor(audio_mix_mags).unsqueeze(0)
		audio_mix_phases = torch.FloatTensor(audio_mix_phases).unsqueeze(0)
		audio_mags = audio_mix_mags.to(opt.device) #dont' care for testing

		#separate for video 1
		log_audio_mags = torch.log(audio_mags + 1e-10)
		visual_features = net_visual(objects_visuals.to(opt.device))
		predicted_masks = gen_unet(log_audio_mags, visual_features)

		reconstructed_signal = get_separated_audio(audio_mix_mags, audio_mix_phases, predicted_masks, opt, log_freq=1)
		sep_audio[-samples_per_window:] = sep_audio[-samples_per_window:] + reconstructed_signal
		overlap_count[-samples_per_window:] = overlap_count[-samples_per_window:] + 1

		#divide the aggregated predicted audio by the overlap count
		separation1 = clip_audio(np.divide(sep_audio, overlap_count) * 2)

		#output original and separated audios
		output_dir = os.path.join('./results/', opt_folder)#os.path.join(opt.output_dir_root, opt.video1_name + 'VS' + opt.video2_name)
		#print(output_dir)
		if not os.path.isdir(output_dir):
			os.mkdir(output_dir)
		#sf.write(os.path.join(output_dir, 'audio_mixed.wav'), audio, opt.audio_sampling_rate, 'PCM_24')
		#sf.write(os.path.join(output_dir, f'audio{i}_separated.wav'), separation1, opt.audio_sampling_rate, 'PCM_24')
		save_wav(os.path.join(output_dir, f'audio_mixed.wav'), audio, opt.audio_sampling_rate)
		save_wav(os.path.join(output_dir, f'audio{i}_separated.wav'), separation1, opt.audio_sampling_rate)
		#save the two detections
		object_image.save(os.path.join(output_dir,  f'audio{i}_det.png'))
		
