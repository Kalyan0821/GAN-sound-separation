import torch
import torchvision
from .networks import Resnet18, AudioVisual7layerUNet, AudioVisual5layerUNet, weights_init, AudioVisualClassifier


def build_visual(pool_type="avgpool", fc_out=512, weights='', pretrained=True):
    original_resnet = torchvision.models.resnet18(pretrained=pretrained)
    if pool_type == "conv1x1":
        net = Resnet18(original_resnet, pool_type=pool_type, with_fc=True, fc_in=6272, fc_out=fc_out)  # 3x224x224 => 512x1x1
    else:
        net = Resnet18(original_resnet, pool_type=pool_type)  # 3x224x224 => 512x1x1

    if len(weights) > 0:
        print("Loading weights for visual stream")
        net.load_state_dict(torch.load(weights))

    return net


def build_unet(unet_num_layers=7, ngf=64, input_channels=1, output_channels=1, with_decoder=True, weights=''):
    if unet_num_layers == 7:
        # 1xFxT, 512x1x1 => 1xFxT
        #                => 1024x(F/128)x(T/128)
        net = AudioVisual7layerUNet(ngf=ngf, input_channels=input_channels, output_channels=output_channels, with_decoder=with_decoder)
    elif unet_num_layers == 5:
        # 1xFxT, 512x1x1 => 1xFxT
        #                => 1024x(F/32)x(T/32)
        net = AudioVisual5layerUNet(ngf=ngf, input_channels=input_channels, output_channels=output_channels, with_decoder=with_decoder)

    net.apply(weights_init)

    if len(weights) > 0:
        print('Loading weights for UNet')
        net.load_state_dict(torch.load(weights))

    return net


def build_classifier(input_channels=1024, pool_type="avgpool"):
    net = AudioVisualClassifier(input_channels=input_channels, pool_type=pool_type)  # 1024xfxt => 1 
    return net
