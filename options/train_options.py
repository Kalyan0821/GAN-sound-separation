from .base_options import BaseOptions


class TrainOptions(BaseOptions):
	def __init__(self):
		super().__init__()
		
		self.parser.add_argument('--display_freq', type=int, default=10, help='frequency of displaying average loss and accuracy')
		self.parser.add_argument('--save_latest_freq', type=int, default=200, help='frequency of saving the latest results')
		self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
		self.parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
		self.parser.add_argument('--decay_factor', type=float, default=0.1, help='decay factor for learning rate')
		self.parser.add_argument('--tensorboard', type=bool, default=False, help='use tensorboard to visualize loss change ')
		self.parser.add_argument('--measure_time', type=bool, default=False, help='measure time of different steps during training')
		self.parser.add_argument('--niter', type=int, default=1, help='# of epochs to train, set to 1 because we are doing random sampling from the whole dataset')
		self.parser.add_argument('--num_batch', default=30000, type=int, help='number of batches to train')
		self.parser.add_argument('--num_object_per_video', default=2, type=int, help='max number of objects detected in a video clip')		
		self.parser.add_argument('--validation_on', type=bool, default=True, help='whether to test on validation set during training')
		self.parser.add_argument('--validation_freq', type=int, default=500, help='frequency of testing on validation set')
		self.parser.add_argument('--validation_batches', type=int, default=10, help='number of batches to test for validation')
		self.parser.add_argument('--validation_visualization', type=bool, default=False, help='whether save validation predictions')
		self.parser.add_argument('--num_visualization_examples', type=int, default=20, help='number of examples to visualize')		
		self.parser.add_argument('--subtract_mean', default=True, type=bool, help='subtract channelwise mean from input image')
		self.parser.add_argument('--preserve_ratio', default=False, type=bool, help='whether boudingbox aspect ratio should be preserved when loading')
		self.parser.add_argument('--enable_data_augmentation', type=bool, default=True, help='whether to augment input audio/image')

		# model arguments
		self.parser.add_argument('--visual_pool', type=str, default='maxpool', help='avg/max pool or using a conv1x1 layer for visual stream feature')
		self.parser.add_argument('--classifier_pool', type=str, default='maxpool', help="avg or max pool for classifier stream feature")
		self.parser.add_argument('--weights_visual', type=str, default='', help="weights for visual stream")
		self.parser.add_argument('--weights_unet', type=str, default='', help="weights for unet")
		self.parser.add_argument('--weights_classifier', type=str, default='', help="weights for audio classifier")
		self.parser.add_argument('--unet_num_layers', type=int, default=7, choices=(5, 7), help="unet number of layers")
		self.parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
		self.parser.add_argument('--unet_input_nc', type=int, default=1, help="input spectrogram number of channels")
		self.parser.add_argument('--unet_output_nc', type=int, default=1, help="output spectrogram number of channels")
		self.parser.add_argument('--number_of_classes', default=15, type=int, help='number of classes')
		self.parser.add_argument('--classifier_loss_weight', default=1, type=float, help='weight for classifier loss')
		self.parser.add_argument('--coseparation_loss_weight', default=1, type=float, help='weight for reconstruction loss')
		self.parser.add_argument('--mask_loss_type', default='L1', type=str, choices=('L1', 'L2', 'BCE'), help='type of reconstruction loss on mask')
		self.parser.add_argument('--weighted_loss', action='store_true', help="weighted loss")
		self.parser.add_argument('--log_freq', type=bool, default=True, help="whether use log-scale frequency")		
		self.parser.add_argument('--mask_thresh', default=0.5, type=float, help='mask threshold for binary mask')
		self.parser.add_argument('--with_additional_scene_image', action='store_true', help="whether to append an extra scene image")	

		# optimizer arguments
		self.parser.add_argument('--lr_visual', type=float, default=0.0001, help='learning rate for visual stream')
		self.parser.add_argument('--lr_unet', type=float, default=0.001, help='learning rate for unet')
		self.parser.add_argument('--lr_classifier', type=float, default=0.001, help='learning rate for audio classifier')
		self.parser.add_argument('--lr_steps', nargs='+', type=int, default=[10000, 20000], help='steps to drop LR in training samples')
		self.parser.add_argument('--optimizer', default='sgd', type=str, help='adam or sgd for optimization')
		self.parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam')
		self.parser.add_argument('--weight_decay', default=0.0001, type=float, help='weights regularizer')
		
		# train/val/test mode
		self.mode = 'train'