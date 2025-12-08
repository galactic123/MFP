import numpy as np
import math
import torch
import torch.nn as nn

from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel



class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        gamma_state = getattr(args, 'gamma_state', 'train')
        if isinstance(gamma_state, bool):
            gamma_state = 'train' if gamma_state else 'infer'
        self.gamma_state = str(gamma_state).lower()
        self.gamma = float(getattr(args, 'gamma', 0.23))
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        
        self.mfp_hidden_size = int(getattr(args, 'mfp_hidden_size', 1024))  

        self.w_q = nn.Linear(self.mfp_hidden_size, self.mfp_hidden_size)
        self.w_k = nn.Linear(self.mfp_hidden_size, self.mfp_hidden_size)
        self.w_v = nn.Linear(self.mfp_hidden_size, self.mfp_hidden_size)
            

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
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.mfp_hidden_size)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

    def get_frequency_subimg(self, image, cutoff_frequency = 30):
        
        bz, channels, rows, cols= image.shape
        assert channels == 3
        def gaussian_lowpass_filter(size, cutoff):
            rows, cols = size
            crow, ccol = rows // 2, cols // 2
            mask = torch.zeros((rows, cols), dtype=torch.float32)
            y, x = torch.meshgrid(torch.arange(rows, dtype=torch.float32), 
                                torch.arange(cols, dtype=torch.float32), indexing='ij')

            d_squared = (x - ccol) ** 2 + (y - crow) ** 2

            mask = torch.exp(-d_squared / (2 * (cutoff ** 2)))

            return mask
        def gaussian_highpass_filter(size, cutoff):
            lowpass_mask = gaussian_lowpass_filter(size, cutoff)
            return 1 - lowpass_mask
        def gaussian_band_filter(size, cutoff):
            lowpass_mask1 = gaussian_lowpass_filter(size, cutoff)
            lowpass_mask2 = gaussian_lowpass_filter(size, cutoff*2)
            return lowpass_mask1 * lowpass_mask2
        def apply_filter(image_tensor, filter_func, cutoff, type='A'):
            image_tensor=image_tensor.to(dtype=torch.float32)
            filtered_image = torch.zeros_like(image_tensor).to(device=image_tensor.device)
            f_image = torch.fft.fft2(image_tensor,dim=(-2,-1))

            fshift = torch.fft.fftshift(f_image).to(device=image_tensor.device)
            filter_mask = filter_func((rows, cols), cutoff).to(device=image_tensor.device)
            if self.gamma_state == 'train':
                seed = torch.ones(bz, 3, 336, 336, device=image_tensor.device)
            else:
                seed = torch.rand(bz, 3, 336, 336, device=image_tensor.device) * self.gamma
            if type=='A':
                fshift_filtered = fshift * filter_mask * seed
            elif type=='B':
                fshift_filtered = fshift * filter_mask * seed
            elif type=='C':
                fshift_filtered = fshift * filter_mask * seed

            f_ishift = torch.fft.ifftshift(fshift_filtered)
            img_back = torch.fft.ifft2(f_ishift,dim=(-2,-1))
            img_back_magnitude = torch.abs(img_back)
            filtered_image = img_back_magnitude.to(dtype=torch.bfloat16)            
            return filtered_image

        lowpass_image_tensor = apply_filter(image, gaussian_lowpass_filter, cutoff_frequency, type='A')
        
        highpass_image_tensor = apply_filter(image, gaussian_highpass_filter, cutoff_frequency,type='B')

        midpass_image_tnesor = apply_filter(image, gaussian_band_filter, cutoff_frequency, type='C')

        return highpass_image_tensor, midpass_image_tnesor, lowpass_image_tensor


    def get_frequency_image(self, image):
        highpass_images, middle_images, lowpass_images = self.get_frequency_subimg(image)
        return highpass_images, middle_images, lowpass_images
    def forward(self, images, highpass_images, lowpass_images):
        if type(images) is list:
            assert False
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            
            highpass_images, middle_images, lowpass_images = self.get_frequency_image(images.to(device=self.device))
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)

            highpass_image_forward_outs = self.vision_tower(highpass_images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            lowpass_image_forward_outs = self.vision_tower(lowpass_images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            
            image_forward_outs = self.feature_select(image_forward_outs)
            highpass_image_forward_outs = self.feature_select(highpass_image_forward_outs)
            lowpass_image_forward_outs = self.feature_select(lowpass_image_forward_outs)
            auxiliary_features = torch.concat((highpass_image_forward_outs.unsqueeze(dim=-1), lowpass_image_forward_outs.unsqueeze(dim=-1)), dim=3).transpose(-1, -2)            
            image_features = torch.transpose(image_forward_outs.unsqueeze(dim=-1), -1, -2)
            image_features.requires_grad_()
            auxiliary_features.requires_grad_()
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
