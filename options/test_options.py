from .base_options import BaseOptions


class TestOptions(BaseOptions):
	def __init__(self):
		super().__init__()

		self.parser.add_argument('--subtract_mean', type=lambda x: (str(x).lower()=="true"), default=True, help="subtract channelwise mean from input image")
		self.parser.add_argument('--preserve_ratio', type=lambda x: (str(x).lower()=="true"), default=False, help="whether bouding box aspect ratio should be preserved when loading")		

		self.parser.add_argument("--visual_pool", type=str, default="conv1x1", help="avgpool/maxpool/conv1x1, for visual stream")
		self.parser.add_argument("--classifier_pool", type=str, default="maxpool", help="avgpool/maxpool, for classifier stream")
		self.parser.add_argument("--weights_visual", type=str, default='', help="weights for visual stream")
		self.parser.add_argument("--weights_unet", type=str, default='', help="weights for unet")
		self.parser.add_argument("--unet_num_layers", type=int, default=7, choices=(5, 7), help="unet number of layers")
		self.parser.add_argument("--unet_ngf", type=int, default=64, help="unet base channel dimension")
		self.parser.add_argument("--unet_input_nc", type=int, default=1, help="input spectrogram number of channels")
		self.parser.add_argument("--unet_output_nc", type=int, default=1, help="output spectrogram number of channels")
		self.parser.add_argument("--number_of_classes", default=15, type=int, help="number of instrument classes")
		self.parser.add_argument("--with_additional_scene_image", action="store_true", help="whether to append an extra scene image")	
		self.parser.add_argument('--softmax_constraint', type=lambda x: (str(x).lower()=="true"), default=True, help="if True, impose SoftMax constraint. If False, use consistency loss")
		self.parser.add_argument('--logscale_freq', type=lambda x: (str(x).lower()=="true"), default=True, help="whether use log-scale frequency")		
		self.parser.add_argument('--epoch', type=str, default="1", help="load weights from this epoch")		
		# train/val/test mode
		self.mode = "test"
