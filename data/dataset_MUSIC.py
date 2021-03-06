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
import time


class MUSICDataset(Dataset):

	def preload(self):

		self.clip_det_dict = dict()
		self.audio_dict = dict()
		self.clean_audio_dict = dict()
		self.clean_detection_dict = dict()

		for video in self.detection_dic:
			for clip_det_path in self.detection_dic[video]:

				self.clip_det_dict[clip_det_path] = np.load(clip_det_path)

				audio_path = get_audio_path_MUSIC(clip_det_path)
				audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate)
				self.audio_dict[audio_path] = (audio, audio_rate)

		for music_label in self.train_solos_dict:
			for clean_detection_path in self.train_solos_dict[music_label]:
				
				clean_audio_path = get_audio_path_MUSIC(clean_detection_path)
				clean_audio, clean_audio_rate = librosa.load(clean_audio_path, sr=self.opt.audio_sampling_rate)
				self.clean_audio_dict[clean_audio_path] = (clean_audio, clean_audio_rate)

				self.clean_detection_dict[clean_detection_path] = np.load(clean_detection_path)	


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

		if self.opt.preload:
			print("Preloading...")
			t1 = time.time()
			self.preload()
			t2 = time.time()
			print(f"Preloading time: {t2-t1}s\n")


	def __len__(self):  # return number of examples (training/validation)
		if self.opt.mode == 'train':
			return self.opt.batchSize * self.opt.num_batch  # number of training examples (default: 32*30000)
		elif self.opt.mode == 'val':
			return self.opt.batchSize * self.opt.validation_batches  # number of validation examples (default: 10*32)


	def __getitem__(self, idx):
		# get 1 random video to separate- "idx" plays no role here
		video = random.sample(self.detection_dic.keys(), 1)[0]
		ground_truth_labels = get_ground_truth_labels_MUSIC(video)		

		clip_det_path = random.choice(self.detection_dic[video])  # randomly sample 1 clip from the video

		if self.opt.preload:
			clip_det_bbs = sample_object_detections(self.clip_det_dict[clip_det_path])  # B x 7 array (B discovered classes in the clip)
		else:
			clip_det_bbs = sample_object_detections(np.load(clip_det_path))

		detected_labels_idx = clip_det_bbs[:, 1].astype(int)  # all B detected class indices
		assert clip_det_bbs.shape[0] != 0  
		assert 0 not in clip_det_bbs[:, 1]  # Make sure __background__ is not a detected class

		objects_visuals = []  # one cropped out BB per discovered class in the clip
		objects_labels = []  # label corresponding to each BB
		objects_audio_mag = []  # audio magnitude spectogram from the clip corresponding to each BB
		objects_audio_phase = []  # audio phase spectogram from the clip corresponding to each BB
		objects_ids = []  # a unique integer id for each clip, repeated over all BBs in the clip
		solos_visuals = []  #  one cropped out BB from the solo corresponding to each discovered class in the clip
		solos_audio_mag = []  # audio magnitude spectogram from a solo corresponding to each BB
		solos_audio_phase = []  # audio phase spectogram from a solo corresponding to each BB

		clip_id = random.randint(1, 100000000000)  # generate a random UNIQUE integer id for each clip
		# audio
		audio_path = get_audio_path_MUSIC(clip_det_path)

		if self.opt.preload:
			audio, audio_rate = self.audio_dict[audio_path]  # load audio of clip at 11025 Hz (default)
		else:
			audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate)


		audio_segment = sample_audio(audio, self.opt.audio_window)  # sample close to 6 secs randomly (default 65535 samples)
		
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

			else:  # No detected label matches with ground truth
				# print("Could not match detected (2) and true labels (1)")
				music_labels_str = ground_truth_labels
				detected_labels_idx = [random.choice(detected_labels_idx)]  # Pick one at random


		elif (len(ground_truth_labels) == 2) and (clip_det_bbs.shape[0] == 1):
			# Include only the available BB
			if ground_truth_labels[0] == self.detector_to_MUSIC_label.get(self.detector_labels[detected_labels_idx[0]]):
				music_labels_str = [ground_truth_labels[0]]

			elif ground_truth_labels[1] == self.detector_to_MUSIC_label.get(self.detector_labels[detected_labels_idx[0]]):
				music_labels_str = [ground_truth_labels[1]]

			else:
				# print("Could not match detected (1) and true labels (2)")
				music_labels_str = [random.choice(ground_truth_labels)]


		elif (len(ground_truth_labels) == 2) and (clip_det_bbs.shape[0] == 2):
			# Map each BB to a ground-truth label
			if (ground_truth_labels[0] == self.detector_to_MUSIC_label.get(self.detector_labels[detected_labels_idx[0]])) or \
			   (ground_truth_labels[1] == self.detector_to_MUSIC_label.get(self.detector_labels[detected_labels_idx[1]])):

				music_labels_str = ground_truth_labels

			elif (ground_truth_labels[1] == self.detector_to_MUSIC_label.get(self.detector_labels[detected_labels_idx[0]])) or \
				 (ground_truth_labels[0] == self.detector_to_MUSIC_label.get(self.detector_labels[detected_labels_idx[1]])):

				music_labels_str = ground_truth_labels[::-1]

			else:
				# print("Could not match detected (2) and true labels (2)")
				music_labels_str = ground_truth_labels[::random.choice([1, -1])]

		assert len(music_labels_str) == len(detected_labels_idx)

		### RANDOM SOLOS ###
		for music_label in music_labels_str:
			clean_detection_path = random.choice(self.train_solos_dict[music_label])

			clean_audio_path = get_audio_path_MUSIC(clean_detection_path)

			if self.opt.preload:
				clean_audio, clean_audio_rate = self.clean_audio_dict[clean_audio_path]
			else:
				clean_audio, clean_audio_rate = librosa.load(clean_audio_path, sr=self.opt.audio_sampling_rate)

			clean_audio_segment = sample_audio(clean_audio, self.opt.audio_window)
			clean_audio_mag, clean_audio_phase = generate_spectrogram_magphase(clean_audio_segment, self.opt.stft_frame, self.opt.stft_hop)
			solos_audio_mag.append(torch.FloatTensor(clean_audio_mag).unsqueeze(0))
			solos_audio_phase.append(torch.FloatTensor(clean_audio_phase).unsqueeze(0))		

			if self.opt.preload:
				clean_detection_bbs = sample_object_detections(self.clean_detection_dict[clean_detection_path])  # B x 7 array
			else:
				clean_detection_bbs = sample_object_detections(np.load(clean_detection_path))  # B x 7 array


			clean_detected_labels_idx = clean_detection_bbs[:, 1].astype(int)  # all B detected class indices
			assert clean_detection_bbs.shape[0] != 0
			assert 0 not in clean_detection_bbs[:, 1]

			if clean_detection_bbs.shape[0] == 2:
				if music_label == self.detector_to_MUSIC_label.get(self.detector_labels[clean_detected_labels_idx[0]]):
					clean_detected_labels_idx = [clean_detected_labels_idx[0]]
				elif music_label == self.detector_to_MUSIC_label.get(self.detector_labels[clean_detected_labels_idx[1]]):
					clean_detected_labels_idx = [clean_detected_labels_idx[1]]
				else:
					# print("SOLOS: Could not match detected (2) and true labels (1)")
					clean_detected_labels_idx = [random.choice(clean_detected_labels_idx)]

			for j in range(clean_detection_bbs.shape[0]):
				if clean_detection_bbs[j, 1].astype(int) not in clean_detected_labels_idx:
					continue
				
				clean_frame_path = os.path.join(get_frames_path_MUSIC(clean_detection_path), str(int(clean_detection_bbs[j, 0])).zfill(6)+".png")
				clean_object_image = Image.open(clean_frame_path).convert('RGB').crop(
					(clean_detection_bbs[j,-4], clean_detection_bbs[j,-3], clean_detection_bbs[j,-2], clean_detection_bbs[j,-1]))

				solos_visuals.append(self.vision_transform(clean_object_image).unsqueeze(0))


		### CURRENT CLIP ###
		for i in range(clip_det_bbs.shape[0]):  # iterate over the B BB-images chosen from the clip

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
			objects_labels.append(self.opt.number_of_classes - 1)
			objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
			objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))
			objects_ids.append(clip_id)

	
		# stack (assume B discovered classes in the clip)
		visuals = np.vstack(objects_visuals)  # all B croppped out BB-images for the clip
		audio_mags = np.vstack(objects_audio_mag)  # overall audio spectograms (repeated for all B BB-images)
		audio_phases = np.vstack(objects_audio_phase)
		labels = np.vstack(objects_labels)  # all B labels in the clip, in [-1, 15] (-1: background, 15: random scene)
		clip_ids = np.vstack(objects_ids)  # all B clip ids (repeated for all BB-images)
		solo_audio_mags = np.vstack(solos_audio_mag)  # all B randomly sampled, solo audio spectograms 
		solo_audio_phases = np.vstack(solos_audio_phase)
		solo_visuals = np.vstack(solos_visuals)


		# Return a dict for each training/validation "example" (index)
		data = {
			"labels": labels,  # B x 1
			"vids": clip_ids,  # B x 1
			"audio_mags": audio_mags,  # B x 1 x F x T (F=512, T=256)
			"solo_audio_mags": solo_audio_mags,  # B x 1 x F x T
			"visuals": visuals,  # B x 3 x n x n (n=224)
			"solo_visuals": solo_visuals  # B x 3 x n x n
			}

		if self.opt.mode in ['val', 'test']:  # for quantitative evaluation, include phase spectograms as well
			data['audio_phases'] = audio_phases  # B x 1 x F x T
			data['solo_audio_phases'] = solo_audio_phases  # B x 1 x F x T

		return data
