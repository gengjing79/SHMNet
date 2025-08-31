from torchvision.models.resnet import resnet50, resnet101
from torchvision.models.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
from torchvision.models.convnext import convnext_tiny, convnext_small, convnext_base
from torchvision.models.efficientnet import efficientnet_v2_s, efficientnet_v2_l
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models.swin_transformer import swin_s, swin_b
from .SHHNet import SHHNet1_1, SHHNet1_2, SHHNet1_3, SHHNet2_1, SHHNet2_2, SHHNet2_3


cfgs = {
    'resnet50': resnet50,
    'resnet101': resnet101,
    'mobilenet_s': mobilenet_v3_small,
    'mobilenet_l': mobilenet_v3_large,
    'convnext_t': convnext_tiny,
    'convnext_s': convnext_small,
    'convnext_b': convnext_base,
    'ef2s': efficientnet_v2_s,
    'ef2l': efficientnet_v2_l,
    'swin_s': swin_s,
    'swin_b': swin_b,
    'vit': vit_b_16,

    'SHMNet1_1': SHHNet1_1,
    'SHMNet1_2': SHHNet1_2,
    'SHMNet1_3': SHHNet1_3,
    'SHMNet2_1': SHHNet2_1,
    'SHMNet2_2': SHHNet2_2,
    'SHMNet2_3': SHHNet2_3,

       }

def calssicmodel_using_name(model_name):
    return cfgs[model_name]

def find_model_using_name(model_name, num_classes):
    return cfgs[model_name](num_classes)