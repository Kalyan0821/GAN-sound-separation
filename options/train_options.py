from .base_options import BaseOptions


class TrainOptions(BaseOptions):
	def __init__(self):
		super().__init__()
		
		self.parser.add_argument("--display_freq", type=int, default=20, help="frequency of displaying average loss and accuracy")
		# self.parser.add_argument("--validation_freq", type=int, default=500, help="frequency of testing on validation set")
		# self.parser.add_argument("--decay_factor", type=float, default=0.1, help="decay factor for learning rate")
		self.parser.add_argument("--tensorboard", type=lambda x: (str(x).lower()=="true"), default=False, help="use tensorboard to visualize loss change")


		# self.parser.add_argument("--measure_time", type=bool, default=False, help="measure time of different steps during training")
		self.parser.add_argument("--num_epochs", type=int, default=5, help="# of training epochs")
		self.parser.add_argument("--num_batch", default=30000, type=int, help="number of batches per training epoch")
		# self.parser.add_argument("--validation_on", type=bool, default=True, help="whether to test on validation set during training")
		# self.parser.add_argument("--validation_batches", type=int, default=10, help="number of batches to test for validation")
		# self.parser.add_argument("--validation_visualization", type=bool, default=False, help="whether save validation predictions")
		# self.parser.add_argument("--num_visualization_examples", type=int, default=20, help="number of examples to visualize")		
		
		self.parser.add_argument("--subtract_mean", type=lambda x: (str(x).lower()=="true"), default=True, help="subtract channelwise mean from input image")
		self.parser.add_argument("--preserve_ratio", type=lambda x: (str(x).lower()=="true"), default=False, help="whether bouding box aspect ratio should be preserved when loading")		
		self.parser.add_argument("--enable_data_augmentation", type=lambda x: (str(x).lower()=="true"), default=True, help="whether to augment input audio/image")

		self.parser.add_argument("--visual_pool", type=str, default="conv1x1", help="avgpool/maxpool/conv1x1, for visual stream")
		self.parser.add_argument("--classifier_pool", type=str, default="maxpool", help="avgpool/maxpool, for classifier stream")
		self.parser.add_argument("--weights_visual", type=str, default='', help="weights for visual stream")
		self.parser.add_argument("--weights_unet", type=str, default='', help="weights for unet")
		self.parser.add_argument("--unet_num_layers", type=int, default=7, choices=(5, 7), help="unet number of layers")
		self.parser.add_argument("--unet_ngf", type=int, default=64, help="unet base channel dimension")
		self.parser.add_argument("--unet_input_nc", type=int, default=1, help="input spectrogram number of channels")
		self.parser.add_argument("--unet_output_nc", type=int, default=1, help="output spectrogram number of channels")
		self.parser.add_argument("--number_of_classes", default=15, type=int, help="number of instrument classes")

		self.parser.add_argument("--softmax_constraint", type=lambda x: (str(x).lower()=="true"), default=True, help="if True, impose SoftMax constraint. If False, use consistency loss")

		self.parser.add_argument("--consistency_loss_weight", default=20, type=float, help="weight for consistency loss")
		self.parser.add_argument("--mask_loss_type", default="L1", type=str, choices=("L1", "L2", "BCE"), help="type of consistency loss on mask")
		# self.parser.add_argument("--weighted_loss", action="store_true", help="weighted loss")

		self.parser.add_argument("--logscale_freq", type=lambda x: (str(x).lower()=="true"), default=True, help="whether use log-scale frequency")
		self.parser.add_argument("--with_additional_scene_image", action="store_true", help="whether to append an extra scene image")	


		self.parser.add_argument("--lr_visual", type=float, default=0.0001, help="learning rate for visual stream")
		self.parser.add_argument("--lr_unet", type=float, default=0.001, help="learning rate for unet")
		self.parser.add_argument("--lr_classifier", type=float, default=0.001, help="learning rate for audio classifier")
		# self.parser.add_argument("--lr_steps", nargs="+", type=int, default=[10000, 20000], help="steps to drop LR in training samples")
		self.parser.add_argument("--optimizer", default="sgd", type=str, choices=("adam", "sgd"), help="optimization algorithm")
		self.parser.add_argument("--beta1", default=0.9, type=float, help="momentum for sgd, beta1 for adam")
		self.parser.add_argument("--weight_decay", default=0.0001, type=float, help="weights regularizer")
		self.parser.add_argument("--num_disc_updates", default=1, type=int, help="number of disc updates before a gen update")

		self.parser.add_argument("--preload", type=lambda x: (str(x).lower()=="true"), default=False, help="whether to preload all data")
		
		# train/val/test mode
		self.mode = "train"