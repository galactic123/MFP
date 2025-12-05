import numpy as np
import math
import torch
import torch.nn as nn

from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
# from .modeling_clip import CLIPVisionModel



class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        

        self.w_q = nn.Linear(1024, 1024)
        self.w_k = nn.Linear(1024, 1024)
        self.w_v = nn.Linear(1024, 1024)
        # self.filter_A = nn.Parameter(torch.rand(3, 336, 336), requires_grad=True)
        # self.filter_B = nn.Parameter(torch.rand(3, 336, 336), requires_grad=True)
        # self.filter_C = nn.Parameter(torch.rand(3, 336, 336), requires_grad=True)
        # self.choice_d = nn.Linear(1024, 1024)
        # self.choice_o = nn.Linear(1024, 1024)

        if not delay_load:
            self.load_model()
        else:
        # self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        # self.vision_tower = CLIPVisionModel(self.cfg_only)
            

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        # self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        # self.vision_tower = CLIPVisionModel(self.cfg_only)
        self.vision_tower.requires_grad_(False)
        
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features


    def forward(self, images, highpass_images, lowpass_images):
        if type(images) is list:
            assert False
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features
           
             
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
