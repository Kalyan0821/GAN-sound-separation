# Test out dataloader and options
from options.train_options import TrainOptions
from data.dataloader import create_dataloader


opt = TrainOptions().parse()
dataloader_train = create_dataloader(opt)

for example in dataloader_train:
	print(example)
	break

