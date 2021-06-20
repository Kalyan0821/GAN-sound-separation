import torch
import torch.nn as nn
import torch.nn.functional as F

def unet_conv(input_channels, output_channels, norm_layer=nn.BatchNorm2d):
    """ c_in x n x n => c_out x n/2 x n/2 """

    downconv = nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1)
    downnorm = norm_layer(output_channels)
    downrelu = nn.LeakyReLU(0.2, True)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_channels, output_channels, outermost=False, norm_layer=nn.BatchNorm2d, no_sigmoid=True):
    """ c_in x n x n => c_out x 2n x 2n """

    upconv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1)
    upnorm = norm_layer(output_channels)
    uprelu = nn.ReLU(True)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        if no_sigmoid:
            return nn.Sequential(*[upconv])  # No sigmoid
        else:
            return nn.Sequential(*[upconv, nn.Sigmoid()])  # Apply sigmoid
        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    """ Regular Conv2D layer with BatchNorm/ReLU optionally """

    model = [nn.Conv2d(input_channels, output_channels, kernel_size=kernel, stride=stride, padding=paddings)]
    if batch_norm:
        model.append(nn.BatchNorm2d(output_channels))
    if Relu:
        model.append(nn.ReLU())

    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class Resnet18(nn.Module):
    def __init__(self, original_resnet, input_channels=3, pool_type='avgpool', with_fc=False, fc_in=512, fc_out=512, custom_first_layer=True):
        super().__init__()
        self.pool_type = pool_type
        self.with_fc = with_fc

        if custom_first_layer:
            # Define custom 1st CONV layer to handle different number of channels for images and spectrograms
            # For all other layers, use ImageNet pre-trained weights
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            layers = [self.conv1]
            layers.extend(
                list(original_resnet.children())[1:-2]  # Discard POOL and SOFT layers
                )
        else:
            # For all layers, use ImageNet pre-trained weights
            layers = list(original_resnet.children())[:-2]

        self.feature_extraction = nn.Sequential(*layers)  # ResNet features before pooling: 512 x 7 x 7
        
        if self.pool_type == 'conv1x1':
            self.conv1x1 = create_conv(input_channels=512, output_channels=128, kernel=1, paddings=0)
            self.conv1x1.apply(weights_init)
        if self.with_fc:
            self.fc = nn.Linear(fc_in, fc_out)
            self.fc.apply(weights_init)

    def forward(self, x):
        """ x: 3 x 224 x 224
            out: 512 x 1 x 1 """
        x = self.feature_extraction(x)  # 512 x 7 x 7

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # 512 x 1 x 1
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1))  # 512 x 1 x 1
        elif self.pool_type == 'conv1x1':
            x = self.conv1x1(x)  # 128 x 7 x 7
        else:
            return x

        if self.with_fc:
            x = x.view(x.shape[0], -1)  # 512 or 6272 (=128*7*7)
            x = self.fc(x)  # 512
            if self.pool_type == 'conv1x1':
                x = x.view(x.shape[0], -1, 1, 1)  # 512 x 1 x 1

        return x

class AudioVisual7layerUNet(nn.Module):
    def __init__(self, ngf=64, input_channels=1, output_channels=1, with_decoder=True, no_sigmoid=True):
        super().__init__()
        self.with_decoder = with_decoder

        # initialize layer weights
        self.audionet_convlayer1 = unet_conv(input_channels, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = unet_conv(ngf * 8, ngf * 8)

        if self.with_decoder:
            self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)
            self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 8)
            self.audionet_upconvlayer3 = unet_upconv(ngf * 16, ngf * 8)
            self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf *4)
            self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
            self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
            self.audionet_upconvlayer7 = unet_upconv(ngf * 2, output_channels,
                                                     outermost=True,
                                                     no_sigmoid=no_sigmoid)

    def forward(self, x, visual_feat):
        """ x: 1 x F x T
            visual_feat: C x 1 x 1 (C=512)
            out: 1 x F x T sigmoid mask """

        audio_conv1feature = self.audionet_convlayer1(x)  # 64 x F/2 x T/2
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)  # 128 x F/4 x T/4
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)  # 256 x F/8 x T/8
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)  # 512 x F/16 x T/16
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)  # 512 x F/32 x T/32
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)  # 512 x F/64 x T/64
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)  # 512 x F/128 x T/128

        # CONCATENATE VISUAL AND TILED-AUDIO FEATURES
        visual_feat = visual_feat.repeat(1, 1, audio_conv7feature.shape[2], audio_conv7feature.shape[3])  # 512 x F/128 x T/128
        audioVisual_feature = torch.cat((visual_feat, audio_conv7feature), dim=1)  # (512+512) x F/128 x T/128
        if not self.with_decoder:
            return audioVisual_feature  # 1024 x F/128 x T/128

        elif self.with_decoder:
            audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)  # 512 x F/64 x T/64 
            audio_upconv2feature = self.audionet_upconvlayer2(
                    torch.cat((audio_upconv1feature, audio_conv6feature), dim=1))  # 1024 x F/64 x T/64 => 512 x F/32 x T/32 
            audio_upconv3feature = self.audionet_upconvlayer3(
                    torch.cat((audio_upconv2feature, audio_conv5feature), dim=1))  # 1024 x F/32 x T/32 => 512 x F/16 x T/16
            audio_upconv4feature = self.audionet_upconvlayer4(
                    torch.cat((audio_upconv3feature, audio_conv4feature), dim=1))  # 1024 x F/16 x T/16 => 256 x F/8 x T/8
            audio_upconv5feature = self.audionet_upconvlayer5(
                    torch.cat((audio_upconv4feature, audio_conv3feature), dim=1))  # 512 x F/8 x T/8 => 128 x F/4 x T/4
            audio_upconv6feature = self.audionet_upconvlayer6(
                    torch.cat((audio_upconv5feature, audio_conv2feature), dim=1))  # 256 x F/4 x T/4 => 64 x F/2 x T/2 
            mask_prediction = self.audionet_upconvlayer7(
                    torch.cat((audio_upconv6feature, audio_conv1feature), dim=1))  # 128 x F/2 x T/2 => 1 x F x T

            return mask_prediction  # 1 x F x T


class AudioVisual5layerUNet(nn.Module):
    def __init__(self, ngf=64, input_channels=1, output_channels=1, with_decoder=True):
        super().__init__()
        self.with_decoder = with_decoder

        # initialize layer weights
        self.audionet_convlayer1 = unet_conv(input_channels, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)

        if self.with_decoder:
            self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)
            self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 4)
            self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
            self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
            self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_channels,
                                                     outermost=True,
                                                     no_sigmoid=no_sigmoid)

    def forward(self, x, visual_feat):
        """ x: 1 x F x T
            visual_feat: C x 1 x 1 (C=512)
            out: 1 x F x T sigmoid mask """

        audio_conv1feature = self.audionet_convlayer1(x)  # 64 x F/2 x T/2
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)  # 128 x F/4 x T/4
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)  # 256 x F/8 x T/8
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)  # 512 x F/16 x T/16
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)  # 512 x F/32 x T/32

        # CONCATENATE VISUAL AND TILED-AUDIO FEATURES
        visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[2], audio_conv5feature.shape[3])  # 512 x F/32 x T/32
        audioVisual_feature = torch.cat((visual_feat, audio_conv5feature), dim=1)  # (512+512) x F/32 x T/32
        if not self.with_decoder:
            return audioVisual_feature  # 1024 x F/32 x T/32

        elif self.with_decoder:
            audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)  # 512 x F/16 x T/16
            audio_upconv2feature = self.audionet_upconvlayer2(
                    torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))  # 1024 x F/16 x T/16 => 256 x F/8 x T/8
            audio_upconv3feature = self.audionet_upconvlayer3(
                    torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))  # 512 x F/8 x T/8 => 128 x F/4 x T/4
            audio_upconv4feature = self.audionet_upconvlayer4(
                    torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))  # 256 x F/4 x T/4 => 64 x F/2 x T/2
            mask_prediction = self.audionet_upconvlayer5(
                    torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))  # 128 x F/2 x T/2 => 1 x F x T

            return mask_prediction  # 1 x F x T


class AudioVisualClassifier(nn.Module):
    def __init__(self, input_channels=1024, pool_type="avgpool"):
        super().__init__()

        self.convlayer1 = unet_conv(input_channels, input_channels*2)
        
        if pool_type == "avgpool":
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif pool_type == "maxpool":
            self.pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.fc1 = nn.Sequential(*[nn.Linear(in_features=input_channels*2, out_features=input_channels//2),
                                   nn.ReLU(inplace=True)])

        self.fc2 = nn.Linear(in_features=input_channels//2, out_features=1)   

    def forward(self, audioVisual_feature):
        """ audioVisual_feature: 1024 x F/128 x T/128 (OR) 1024 x F/32 x T/32
            out: 1 """

        x = self.convlayer1(audioVisual_feature)  # 2048 x F/256 x T/256
        x = self.pool(x)  # 2048 x 1 x 1
        x = x.view(x.shape[0], -1)  # 2048
        x = self.fc1(x)  # 512
        out = self.fc2(x)  # 1 (binary classification logit)
        return out
