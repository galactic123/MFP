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
        self.theta_state = getattr(args, 'theta_state', True)
        

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
    def get_frequency_subimg(self, image, cutoff_frequency = 30):
        # 高斯低通滤波器
        # image=image.permute(1, 2, 0)
        
        bz, channels, rows, cols= image.shape
        assert channels == 3
        def gaussian_lowpass_filter(size, cutoff):
            rows, cols = size
            crow, ccol = rows // 2, cols // 2
            mask = torch.zeros((rows, cols), dtype=torch.float32)
            # 创建坐标网格
            y, x = torch.meshgrid(torch.arange(rows, dtype=torch.float32), 
                                torch.arange(cols, dtype=torch.float32), indexing='ij')

            # 计算与中心点的距离
            d_squared = (x - ccol) ** 2 + (y - crow) ** 2

            # 计算高斯分布
            mask = torch.exp(-d_squared / (2 * (cutoff ** 2)))

            return mask
        # 高斯高通滤波器
        def gaussian_highpass_filter(size, cutoff):
            lowpass_mask = gaussian_lowpass_filter(size, cutoff)
            return 1 - lowpass_mask  # 高通滤波器是低通的反转
        def gaussian_band_filter(size, cutoff):
            lowpass_mask1 = gaussian_lowpass_filter(size, cutoff)
            lowpass_mask2 = gaussian_lowpass_filter(size, cutoff*2)
            return lowpass_mask1 * lowpass_mask2  # 高通滤波器是低通的反转            
        # 对每个通道应用傅里叶变换，滤波，然后逆变换
        def apply_filter(image_tensor, filter_func, cutoff, type='A'):
            image_tensor=image_tensor.to(dtype=torch.float32)
            filtered_image = torch.zeros_like(image_tensor).to(device=image_tensor.device)
            f_image = torch.fft.fft2(image_tensor,dim=(-2,-1)) # 3, 336, 336

            fshift = torch.fft.fftshift(f_image).to(device=image_tensor.device)
            filter_mask = filter_func((rows, cols), cutoff).to(device=image_tensor.device)
            # print(image_tensor.size())
            if self.theta_state:
                seed = torch.ones(bz, 3, 336, 336, device=image_tensor.device)
            else:
                seed = torch.rand(bz, 3, 336, 336, device=image_tensor.device) * 0.23
            # print(seed)
            # seedA = torch.rand(bz, 3, 336, 336).to(device=image_tensor.device) * 0.3
            # seedB = torch.rand(bz, 3, 336, 336).to(device=image_tensor.device) * 0.4

            # seed = torch.ones(bz, 3, 336, 336).to(device=image_tensor.device) 
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

            # for c in range(channels):
            #     # 对每个通道进行傅里叶变换
            #     f_image = torch.fft.fft2(image_tensor[:, :, c])
            #     fshift = torch.fft.fftshift(f_image).to(device=image_tensor.device)  # 将低频移到中心
                
            #     # 创建高斯滤波器
            #     filter_mask = filter_func((rows, cols), cutoff).to(device=image_tensor.device)
                
            #     # 应用滤波器
            #     if type=='A':
            #         fshift_filtered = fshift * filter_mask * self.filter_A[:, :, c]
            #     elif type=='B':
            #         fshift_filtered = fshift * filter_mask * self.filter_B[:, :, c]
            #     elif type=='C':
            #         fshift_filtered = fshift * filter_mask * self.filter_C[:, :, c]
                
            #     # 逆傅里叶变换
            #     f_ishift = torch.fft.ifftshift(fshift_filtered)
            #     img_back = torch.fft.ifft2(f_ishift)
            #     img_back_magnitude = torch.abs(img_back)
                
            #     # 存储滤波后的图像
            #     filtered_image[:, :, c] = img_back_magnitude
            # # filtered_image=filtered_image.permute(2, 0, 1)
            # return filtered_image.to(dtype=torch.bfloat16)

        # 应用高斯低通滤波器
        # lowpass_image_tensor = apply_filter(image, gaussian_lowpass_filter, cutoff_frequency, type='A')
        lowpass_image_tensor = apply_filter(image, gaussian_lowpass_filter, cutoff_frequency, type='A')
        
        # 应用高斯高通滤波器
        highpass_image_tensor = apply_filter(image, gaussian_highpass_filter, cutoff_frequency,type='B')

        midpass_image_tnesor = apply_filter(image, gaussian_band_filter, cutoff_frequency, type='C')

        return highpass_image_tensor, midpass_image_tnesor, lowpass_image_tensor


    def get_frequency_image(self, image):
        # highpass_images, middle_images, lowpass_images = [],[],[]
        # for i in range(image.size(0)):
        highpass_images, middle_images, lowpass_images = self.get_frequency_subimg(image)
        #     highpass_images.append(highpass_image)
        #     middle_images.append(middle_image)
        #     lowpass_images.append(lowpass_image)
        # highpass_images = torch.stack(highpass_images)
        # middle_images = torch.stack(middle_images)
        # lowpass_images = torch.stack(lowpass_images)
        return highpass_images, middle_images, lowpass_images
        # assert False
    # @torch.no_grad()
    def forward(self, images, highpass_images, lowpass_images):
        # print(images.size(), highpass_images.size(), lowpass_images.size())
        # highpass_images, middle_images, lowpass_images = self.get_frequency_image(images)
        # print(images.size(), middle_images.size(), highpass_images.size(), lowpass_images.size())
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
            # band_image_forward_outs = self.vision_tower(middle_images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            lowpass_image_forward_outs = self.vision_tower(lowpass_images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            



            image_forward_outs = self.feature_select(image_forward_outs)
            highpass_image_forward_outs = self.feature_select(highpass_image_forward_outs)
            # band_image_forward_outs = self.feature_select(band_image_forward_outs)
            lowpass_image_forward_outs = self.feature_select(lowpass_image_forward_outs)
            # lowpass_image_forward_outs = torch.zeros_like(lowpass_image_forward_outs)
            # highpass_image_forward_outs = torch.zeros_like(highpass_image_forward_outs)
            auxiliary_features = torch.concat((highpass_image_forward_outs.unsqueeze(dim=-1), lowpass_image_forward_outs.unsqueeze(dim=-1)), dim=3).transpose(-1, -2)            
            image_features = torch.transpose(image_forward_outs.unsqueeze(dim=-1), -1, -2)
            # print(image_features.size(), auxiliary_features.size())  
            image_features.requires_grad_()
            auxiliary_features.requires_grad_()
            # image_features = self.choice_d(self.deepfuse(image_features, auxiliary_features)) + self.choice_o(image_features)
            image_features = self.deepfuse(image_features, auxiliary_features) + image_features
            image_features = torch.squeeze(image_features, dim=2)

            # image_features = self.vision_tower(image_features, output_hidden_states=True, begin_layer=6, end_layer=None, use_pre=False)
            # image_features = self.feature_select(image_features).to(images.dtype)
            # highpass_image_features = self.feature_select(highpass_image_forward_outs).to(images.dtype).unsqueeze(dim=-1)
            
            # lowpass_image_features = self.feature_select(lowpass_image_forward_outs).to(images.dtype).unsqueeze(dim=-1)

           
             
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
