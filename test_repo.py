# Test out dataloader and options
from options.train_options import TrainOptions
from data.dataloader import create_dataloader
from data.dataset_MUSIC import MUSICDataset

opt = TrainOptions().parse()
dataloader_train = create_dataloader(opt)
dataset_train = MUSICDataset(opt)

single_sample = dataset_train[0]
for k in single_sample:
	print(k, single_sample[k].shape, single_sample[k].dtype)





#for example in dataloader_train:
#	print(example)
#	break

