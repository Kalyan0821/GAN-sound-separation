# Test out dataloader and options
from options.train_options import TrainOptions
from data.dataloader import create_dataloader
from data.dataset_MUSIC import MUSICDataset

opt = TrainOptions().parse()
opt.seed = 42

# Check dataset
dataset_train = MUSICDataset(opt)
print("\nChecking dataset")
for i, single_sample_dict in enumerate(dataset_train):
	for k in single_sample_dict:
		print(k, single_sample_dict[k].shape, single_sample_dict[k].dtype)
	print()
	if i+1 == 50:
		break


# # Check dataloader
# dataloader_train = create_dataloader(opt)
# print("\nChecking dataloader")
# for i, batch_dict in enumerate(dataloader_train):
# 	for k in batch_dict:
# 		print(k, batch_dict[k].shape, batch_dict[k].dtype)
# 	print()
# 	if i+1 == 5:
# 		break
# 
# 
