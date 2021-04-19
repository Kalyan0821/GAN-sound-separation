import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import os
import h5py
from PIL import Image
from .dataset_utils import get_vid_path_MUSIC, get_audio_path_MUSIC, get_frames_path_MUSIC, get_ground_truth_labels_MUSIC
from .dataset_utils import sample_object_detections, sample_audio, augment_audio, generate_spectrogram_magphase, augment_image
import numpy as np
import librosa


class MUSICDataset(Dataset):
	def __init__(self, opt):
		super().__init__()

		self.opt = opt
		random.seed(opt.seed)

		self.detection_dic = dict()  # {video_name: [clip_detection_npy_paths]} dict

		if opt.mode == "train":
			detections_file = os.path.join(opt.all_paths_dir, opt.dataset, "train.txt")
			train_solos_file = os.path.join(opt.all_paths_dir, opt.dataset, "train_solos.txt")  # single instrument videos
		elif opt.mode == "val":
			detections_file = os.path.join(opt.all_paths_dir, opt.dataset, "val.txt")
		
		with open(detections_file, 'r') as f:
			detections = [s[:-1] for s in f.readlines()]  # list of all .npy clip detection file paths (strip the '\n')
			for detection in detections:  # iterate through all .npy paths
				vid_path = get_vid_path_MUSIC(detection)  # get name of video the clip belongs to
				if vid_path in self.detection_dic:
					self.detection_dic[vid_path].append(detection)
				else:
					self.detection_dic[vid_path] = [detection]

		if opt.mode == "train":
			self.train_solos_dict = dict()  # {video_name: [clip_detection_npy_paths]} dict
			
			with open(train_solos_file, 'r') as f:
				detections = [s[:-1] for s in f.readlines()]  # list of all .npy clip detection file paths (strip the '\n')
				for detection in detections:  # iterate through all .npy paths
					vid_path = get_vid_path_MUSIC(detection)  # get name of video the clip belongs to
					true_label = get_ground_truth_labels_MUSIC(vid_path)
					assert len(true_label) == 1
					true_label = true_label[0] 

					if true_label in self.train_solos_dict:
						self.train_solos_dict[true_label].append(detection)
					else:
						self.train_solos_dict[true_label] = [detection]


		# vision transforms
		if opt.mode == 'val':
			vision_transform_list = [transforms.Resize((224,224)), transforms.ToTensor()]
		elif opt.preserve_ratio:
			vision_transform_list = [transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor()]
		else:
			vision_transform_list = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.ToTensor()]
		if opt.subtract_mean:
			vision_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

		self.vision_transform = transforms.Compose(vision_transform_list)

		# load hdf5 file of scene images
		if opt.with_additional_scene_image:
			h5f_path = os.path.join(opt.scene_path)
			with h5py.File(h5f_path, 'r') as f:
				self.scene_images = f["image"][:]

		self.detector_labels = ['__background__',
						   'Banjo', 'Cello', 'Drum', 'Guitar',
						   'Harp', 'Harmonica', 'Oboe', 'Piano',
						   'Saxophone', 'Trombone', 'Trumpet', 'Violin',
						   'Flute', 'Accordion', 'Horn']


		self.detector_to_MUSIC_label = {
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


	def __len__(self):  # return number of examples (training/validation)
		if self.opt.mode == 'train':
			return self.opt.batchSize * self.opt.num_batch  # number of training examples (default: 30000*32)
		elif self.opt.mode == 'val':
			return self.opt.batchSize * self.opt.validation_batches  # number of validation examples (default: 10*32)


	def __getitem__(self, idx):
		# get 1 random video to separate- "idx" plays no role here
		video = random.sample(self.detection_dic.keys(), 1)[0]
		ground_truth_labels = get_ground_truth_labels_MUSIC(video)		

		clip_det_path = random.choice(self.detection_dic[video])  # randomly sample 1 clip from the video
		clip_det_bbs = sample_object_detections(np.load(clip_det_path))  # C x 7 array (C discovered classes in the clip)
		detected_labels_idx = clip_det_bbs[:, 1].astype(int)  # all C detected class indices
		assert clip_det_bbs.shape[0] != 0  # Make sure __background__ is not a detected class
		assert 0 not in clip_det_bbs[:, 1]

		objects_visuals = []  # one cropped out BB per discovered class in any clip
		objects_labels = []  # label corresponding to each BB
		objects_audio_mag = []  # audio magnitude spectogram from the clip corresponding to each BB
		objects_audio_phase = []  # audio phase spectogram from the clip corresponding to each BB
		objects_ids = []

		clip_id = random.randint(1, 100000000000)  # generate a random UNIQUE integer id for each clip
		# audio
		audio_path = get_audio_path_MUSIC(clip_det_path)
		audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate)  # load audio of clip at 11025 Hz (default)
		audio_segment = sample_audio(audio, self.opt.audio_window)  # load close to 6 secs randomly (default 65535 samples)
		
		if(self.opt.enable_data_augmentation and self.opt.mode == 'train'):  
			audio_segment = augment_audio(audio_segment)

		audio_mag, audio_phase = generate_spectrogram_magphase(audio_segment, self.opt.stft_frame, self.opt.stft_hop)

		# Sample appropriate clean audio segments
		# This step is not perfect, could be modified further
		
		if (len(ground_truth_labels) == 1) and (clip_det_bbs.shape[0] == 1):
			music_labels_str = ground_truth_labels

		elif (len(ground_truth_labels) == 1) and (clip_det_bbs.shape[0] == 2):
			# Include only the correct BB
			if ground_truth_labels[0] == self.detector_to_MUSIC_label.get(self.detector_labels[detected_labels_idx[0]]):
				music_labels_str = ground_truth_labels
				detected_labels_idx = [detected_labels_idx[0]]

			elif ground_truth_labels[0] == self.detector_to_MUSIC_label.get(self.detector_labels[detected_labels_idx[1]]):
				music_labels_str = ground_truth_labels
				detected_labels_idx = [detected_labels_idx[1]]

			else:
				print(ground_truth_labels, detected_labels_idx)
				raise Exception("Could not match detected and true labels\n")

		elif (len(ground_truth_labels) == 2) and (clip_det_bbs.shape[0] == 1):
			# Include only the available BB
			if ground_truth_labels[0] == self.detector_to_MUSIC_label.get(self.detector_labels[detected_labels_idx[0]]):
				music_labels_str = [ground_truth_labels[0]]

			elif ground_truth_labels[1] == self.detector_to_MUSIC_label.get(self.detector_labels[detected_labels_idx[0]]):
				music_labels_str = [ground_truth_labels[1]]

			else:
				print(ground_truth_labels, detected_labels_idx)
				raise Exception("Could not match detected and true labels\n")


		elif (len(ground_truth_labels) == 2) and (clip_det_bbs.shape[0] == 2):
			# Map each BB to a ground-truth label
			if (ground_truth_labels[0] == self.detector_to_MUSIC_label.get(self.detector_labels[detected_labels_idx[0]])) or \
			   (ground_truth_labels[1] == self.detector_to_MUSIC_label.get(self.detector_labels[detected_labels_idx[1]])):

				music_labels_str = ground_truth_labels

			elif (ground_truth_labels[1] == self.detector_to_MUSIC_label.get(self.detector_labels[detected_labels_idx[0]])) or \
				 (ground_truth_labels[0] == self.detector_to_MUSIC_label.get(self.detector_labels[detected_labels_idx[1]])):

				music_labels_str = ground_truth_labels[::-1]

			else:
				print(ground_truth_labels, detected_labels_idx)
				raise Exception("Could not match detected and true labels\n")	

		print(music_labels_str, detected_labels_idx)
		assert len(music_labels_str) == len(detected_labels_idx)

		for music_label in music_labels_str:
			clean_detection_path = random.choice(self.train_solos_dict[music_label])
			clean_audio_path = get_audio_path_MUSIC(clean_detection_path)
			clean_audio, clean_audio_rate = librosa.load(clean_audio_path, sr=self.opt.audio_sampling_rate)
			clean_audio_segment = sample_audio(clean_audio, self.opt.audio_window)
			clean_audio_mag, clean_audio_phase = generate_spectrogram_magphase(clean_audio_segment, self.opt.stft_frame, self.opt.stft_hop)
			pass




		for i in range(clip_det_bbs.shape[0]):  # iterate over the C BB-images chosen from the clip

			if clip_det_bbs[i, 1].astype(int) not in detected_labels_idx:  # don't consider wrong detections
				continue 

			# get path of the single randomly sampled frame of the ith BB-image
			frame_path = os.path.join(get_frames_path_MUSIC(clip_det_path), str(int(clip_det_bbs[i, 0])).zfill(6)+".png")			
			# Crop out the BB image from the sampled frame for each discovered class in the clip
			object_image = Image.open(frame_path).convert('RGB').crop(
				(clip_det_bbs[i,-4], clip_det_bbs[i,-3], clip_det_bbs[i,-2], clip_det_bbs[i,-1]))

			if(self.opt.enable_data_augmentation and self.opt.mode == 'train'):
				object_image = augment_image(object_image)

			# reshape and normalize each BB image
			objects_visuals.append(self.vision_transform(object_image).unsqueeze(0))

			label = clip_det_bbs[i, 1] - 1  # convert class label to zero-based index, i.e., [0, 15] => [-1, 14]
			objects_labels.append(label)
			# store an identical copy of the audio spectogram for each BB image
			objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
			objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))
			objects_ids.append(clip_id)  # to identify which BB image/spectrogram corresponds to which clip (among a batch of examples)
		
		# additional random scene BB image for each video
		if self.opt.with_additional_scene_image:
			scene_image_path = random.choice(self.scene_images)
			scene_image = Image.open(scene_image_path).convert('RGB')
			if(self.opt.enable_data_augmentation and self.opt.mode == 'train'):
				scene_image = augment_image(scene_image)
			objects_visuals.append(self.vision_transform(scene_image).unsqueeze(0))
			objects_labels.append(self.opt.number_of_classes - 1)  ################### Potential mistake: should be 15, NOT 15-1=14
			objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
			objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))
			objects_ids.append(clip_id)
	
		# stack (assume C discovered classes in the clip)
		visuals = np.vstack(objects_visuals)  # all C croppped out BB-images for the clip
		audio_mags = np.vstack(objects_audio_mag)  # all C audio spectograms (repeated for all BB-images)
		audio_phases = np.vstack(objects_audio_phase)
		labels = np.vstack(objects_labels)  # all C labels in the clip, in [-1, 15] (-1: background, 15: random scene)
		clip_ids = np.vstack(objects_ids)  # all C clip ids (repeated for all BB-images)

		# Return a dict for each training/validation "example" (index)
		data = {
			'labels': labels,
			'audio_mags': audio_mags,
			'vids': clip_ids,
			'visuals': visuals
			}
		if self.opt.mode in ['val', 'test']:  # for quantitative evaluation, include phase spectograms as well
			data['audio_phases'] = audio_phases

		return data

