import torch
import torch.nn as nn
import os
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

intermediate = {}
def get_interm(name):
    def hook(model, input, output):
        intermediate[name] = output.detach()
    return hook

class Swin_UNETR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        img_size = cfg['data_size']
        in_channels = cfg['in_channels']
        out_channels = cfg['out_channels']
        feature_size = cfg['feature_size']
        drop_rate = cfg.get('drop_rate', 0)
        attn_drop_rate = cfg.get('attn_drop_rate', 0)
        use_checkpoint = cfg.get('use_checkpoint', True)
        pretrained = cfg.get('pretrained', False)
        self.out_fmaps = cfg['out_fmaps']

        
        if pretrained:
            self.swin_unetr = SwinUNETR(
                img_size=img_size,
                in_channels=1,
                out_channels=14,
                feature_size=48,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                use_checkpoint=use_checkpoint,
            )
            model_dict = torch.load(os.path.join('pretrained_models', 'ssl_pretrained.pt'))["state_dict"]
            self.swin_unetr.load_state_dict(model_dict)
            print("Use pretrained weights")
        else:
            self.swin_unetr = SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=feature_size,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                use_checkpoint=use_checkpoint,
            )
        
        # freeze layers that we don't need
        for i, (name, param) in enumerate(self.swin_unetr.named_parameters()):
            if name.startswith('decoder') or name.startswith('out'):
                param.requires_grad = False
        
        self.swin_unetr.encoder1.register_forward_hook(get_interm('P0'))
        self.swin_unetr.encoder2.register_forward_hook(get_interm('P1'))
        self.swin_unetr.encoder3.register_forward_hook(get_interm('P2'))
        self.swin_unetr.encoder4.register_forward_hook(get_interm('P3'))
        self.swin_unetr.encoder10.register_forward_hook(get_interm('P4'))
        
        
    def forward(self, x):
        outputs = {}
        x = self.swin_unetr(x)
        for k, v in intermediate.items():
            if k in self.out_fmaps:
                outputs[k] = v
        return outputs