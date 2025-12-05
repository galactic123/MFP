import numpy as np
import cv2
import math
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


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

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
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






    def deepfuse(self, image_features, auxiliary_features):
        Q = self.w_q(image_features)
        K = self.w_k(auxiliary_features)
        V = self.w_v(auxiliary_features)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(1024)

        # 应用Softmax进行归一化
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 加权求和
        output = torch.matmul(attention_weights, V)
        # print(output.size())
        return output

        # assert False
    # @torch.no_grad()
    def forward(self, images, highpass_images=None, lowpass_images=None):
        # print(images.size(), highpass_images.size(), lowpass_images.size())
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # highpass_images, lowpass_images = self.get_frequency_image(images)
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)

            highpass_image_forward_outs = self.vision_tower(highpass_images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            lowpass_image_forward_outs = self.vision_tower(lowpass_images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)

            image_features = self.feature_select(image_forward_outs).to(images.dtype).unsqueeze(dim=-1)
            highpass_image_features = self.feature_select(highpass_image_forward_outs).to(images.dtype).unsqueeze(dim=-1)
            
            lowpass_image_features = self.feature_select(lowpass_image_forward_outs).to(images.dtype).unsqueeze(dim=-1)

            auxiliary_features = torch.concat((highpass_image_features, lowpass_image_features), dim=3).transpose(-1, -2)
            image_features = torch.transpose(image_features, -1, -2)
            # print(image_features.size(), auxiliary_features.size())  
            image_features.requires_grad_()
            auxiliary_features.requires_grad_()
            print("------")
            print(self.w_q.weight)
            image_features = self.deepfuse(image_features, auxiliary_features) + image_features            
            image_features = torch.squeeze(image_features, dim=2) 
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
