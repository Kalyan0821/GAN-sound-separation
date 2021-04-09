from torch.utils.data import DataLoader
from .dataset_MUSIC import MUSICDataset  # ".file": "file" is in same directory 

def create_dataloader(opt):
	# create the dataset
	assert opt.dataset in ["MUSIC", "FAIR-Play", "AudioSet"]

	if opt.dataset == "MUSIC":
		dataset = MUSICDataset(opt)
	elif opt.dataset == "FAIR-Play":
		dataset = FairPlayDataset(opt)
	elif opt.dataset == "AudioSet":
		dataset = AudioSetDataset(opt)

	print(f"Dataset for {opt.dataset} was created\n\n")


	# create the dataloader
	if opt.mode == "train":
		dataloader = DataLoader(dataset=dataset,
					batch_size=opt.batchSize,
					shuffle=False,
					num_workers=int(opt.nThreads),
					collate_fn=None)  # define this in utils later

	elif opt.mode == "val":
		dataloader = DataLoader(dataset=dataset,
					batch_size=opt.batchSize,
					shuffle=False,
					num_workers=2,
					collate_fn=None)  # define this in utils later		

	return dataloader




