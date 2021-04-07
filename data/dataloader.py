from torch.utils.data import DataLoader
from dataset import AudioVisualDataset

def create_dataloader(opt):
	# create the dataset
	assert opt.model in ["MUSIC", "FAIR-Play", "AudioSet"]

	dataset = AudioVisualMUSICDataset(opt)  # define __init__() to mimic inititalize()
	print(f"Dataset for {opt.model} was created")

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




