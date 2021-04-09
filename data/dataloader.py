from torch.utils.data import DataLoader
from .dataset_MUSIC import MUSICDataset  # ".file": "file" is in same directory 
from .dataset_utils import object_collate

def create_dataloader(opt):
	# create the dataset
	assert opt.dataset in ["MUSIC", "FAIR-Play", "AudioSet"]

	if opt.dataset == "MUSIC":
		dataset = MUSICDataset(opt)
	elif opt.dataset == "FAIR-Play":
		dataset = FairPlayDataset(opt)
	elif opt.dataset == "AudioSet":
		dataset = AudioSetDataset(opt)

	print(f"Dataset for {opt.dataset} was created")

	# create the dataloader
	if opt.mode == "train":
		dataloader = DataLoader(dataset=dataset,
					batch_size=opt.batchSize,
					shuffle=False,
					num_workers=int(opt.nThreads),
					collate_fn=object_collate)  # custom function that converts any list of examples into stacked tensors
												# can't use default since the 0th dimension (total # BBs) may vary across examples 

	elif opt.mode == "val":
		dataloader = DataLoader(dataset=dataset,
					batch_size=opt.batchSize,
					shuffle=False,
					num_workers=2,
					collate_fn=object_collate)	

	return dataloader




