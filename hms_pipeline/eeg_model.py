import torch.nn as nn
import torch
import timm

import pytorch_lightning as pl
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

class EegInferModule(pl.LightningModule):
    def __init__(self, model, config, verbose=False):
        super().__init__()
        self.model = model
        self.config = config
        if config["use_ema"]:
            ema_decay = config["ema_decay"]
            self.ema_model = AveragedModel(
                model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)
            )
            if verbose:
                print("Using EMA model with decay", ema_decay)

    def forward(self, x):
        if self.config["use_ema"]:
            return self.ema_model(x)
        return self.model(x)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return torch.nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'
    


def gen_coord_tensor(h,w):
    coord_tensor = torch.zeros((1, h, w, 2), dtype=torch.float32)
    h_range = torch.linspace(-1, 1, steps=h)
    w_range = torch.linspace(-1, 1, steps=w)
    coord_tensor[0, :, :, 0] = w_range.unsqueeze(0).expand(h, w)
    coord_tensor[0, :, :, 1] = h_range.unsqueeze(1).expand(h, w)
    return coord_tensor

    
class SpecModel(torch.nn.Module):
    def __init__(self, backbone, pretrained=True, global_pool = {'avg'}, dropout=0.5, hidden_size=16, img_size=[512,512]):
        super().__init__()
        self.model = timm.create_model(backbone, pretrained=pretrained, in_chans=1, )
        # remove classifier and pooling
        # self.model.reset_classifier(0, '')
        total_features = 0
        self.global_pool = global_pool
        self.model.reset_classifier(0, '')
        feature_map = self.model.forward_features(torch.randn(1, 1, img_size[0], img_size[1]))
        in_features = feature_map.shape[1]


        if "attn" in global_pool:
            print("Attn pool features shape", feature_map.shape, "channels", in_features)
            self.attn_pool = timm.layers.AttentionPool2d(in_features, feat_size= (feature_map.shape[2], feature_map.shape[3]))
            total_features += in_features

        if "max" in global_pool:
            self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
            total_features += in_features

        if "avg" in global_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            total_features += in_features

        if "gem" in global_pool:
            self.gem_pool = GeM()
            total_features += in_features

        assert total_features > 0, "At least one global pool should be selected"

        self.classifier = nn.Sequential(
            nn.LayerNorm(total_features),
            nn.Dropout(dropout),
            nn.Linear(total_features, 6),
        )

    def forward(self, x):
        feature_map = self.model.forward_features(x)

        features = []
        if "attn" in self.global_pool:
            attn_features = self.attn_pool(feature_map)
            features.append(attn_features)
            
        if "max" in self.global_pool:
            max_features = self.max_pool(feature_map)
            max_features = max_features.view(max_features.size(0), -1)
            features.append(max_features)
        if "avg" in self.global_pool:
            avg_features = self.avg_pool(feature_map)
            avg_features = avg_features.view(avg_features.size(0), -1)
            features.append(avg_features)

        if "gem" in self.global_pool:
            gem_features = self.gem_pool(feature_map)
            gem_features = gem_features.view(gem_features.size(0), -1)
            features.append(gem_features)
        # print(attn_features.shape, max_features.shape)
        features = torch.cat(features, dim=1)

        output = self.classifier(features)
        return output

class SpecVitModel(torch.nn.Module):
    def __init__(self, backbone, vit_model = "vit_small_patch16_224", img_size = [512,512], pretrained=True, global_pool = 'avg', dropout=0.5, hidden_size=16, feature_layer = [1]):
        super().__init__()

        print("Using model", vit_model)
        self.model = timm.create_model(vit_model, pretrained=pretrained, in_chans=1, drop_rate=dropout, img_size=img_size[0])
        self.model.reset_classifier(6)

    def forward(self, x):
        output = self.model(x)
        return output