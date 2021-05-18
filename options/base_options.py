import argparse
import os
import torch

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

		self.parser.add_argument('--dataset', type=str, default="MUSIC", choices=("MUSIC", "FAIR-Play", "AudioSet"), help='affects how data is loaded')
		self.parser.add_argument('--experiment_id', type=str, help='name of the experiment. It decides where to store models')

		self.parser.add_argument('--all_paths_dir', default='./all_paths')
		# self.parser.add_argument('--scene_path', default='/your_root/hdf5/ADE.h5', help='path to scene images')
		self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
		self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		self.parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
		self.parser.add_argument('--nThreads', default=16, type=int, help='# threads for loading data')
		self.parser.add_argument('--seed', default=0, type=int, help='random seed')

		self.parser.add_argument('--audio_window', default=65535, type=int, help='audio segment length (# samples)')
		self.parser.add_argument('--audio_sampling_rate', default=11025, type=int, help='audio sampling rate')
		self.parser.add_argument('--stft_frame', default=1022, type=int, help="stft frame length")
		self.parser.add_argument('--stft_hop', default=256, type=int, help="stft hop length")

	def parse(self):
		self.opt = self.parser.parse_args()
		# Set 2 additional atributes
		self.opt.mode = self.mode
		self.opt.gpu_ids = [int(str_id) for str_id in self.opt.gpu_ids.split(',') if int(str_id)>=0]

		# Set first GPU as current device
		if len(self.opt.gpu_ids) > 0:
			torch.cuda.set_device(self.opt.gpu_ids[0])

		if self.opt.with_additional_scene_image:
			# 15 non-background classes + 1 random scene class
			# desired_indices: -1=background, [0, 14]=non-background, 15=random-scene
			opt.number_of_classes = opt.number_of_classes + 1   

		self.save()
		return self.opt


	def save(self):
		# Print options
		args = vars(self.opt)  # Gives a dict
		print("------------ Options -------------")
		for k, v in sorted(args.items()):
			print(f"{k}: {v}")
		print('-------------- End ----------------')

		# Save options to disk
		experiment_dir = os.path.join(self.opt.checkpoints_dir, self.opt.experiment_id)
		if not os.path.isdir(experiment_dir):
			os.makedirs(experiment_dir)

		file_name = os.path.join(experiment_dir, "opt.txt")
		with open(file_name, 'wt') as opt_file:
			for k, v in sorted(args.items()):
				opt_file.write(f"{k}: {v}\n")
