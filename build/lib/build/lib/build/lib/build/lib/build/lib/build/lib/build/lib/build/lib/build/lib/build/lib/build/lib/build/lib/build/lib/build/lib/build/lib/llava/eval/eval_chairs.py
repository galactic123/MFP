import argparse
import json
import os
import random
from tkinter import Image
from copy import deepcopy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from llava.constants import *
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.eval.eval_data_loader import COCODataSet
from llava.utils import disable_torch_init
# from llava.eval.model_chairs_loader import ModelLoader
from tqdm import tqdm
from llava.conversation import conv_templates

def get_frequency_subimg(image, cutoff_frequency = 30):
    # 高斯低通滤波器
    # print(image.size())
    image=image.permute(1, 2, 0)
    
    rows, cols, channels= image.shape
    assert channels == 3
    def gaussian_lowpass_filter(size, cutoff):
        rows, cols = size
        crow, ccol = rows // 2, cols // 2
        mask = torch.zeros((rows, cols), dtype=torch.float32).to(device=image.device)
            # 创建坐标网格
        y, x = torch.meshgrid(torch.arange(rows, dtype=torch.float32), 
                            torch.arange(cols, dtype=torch.float32), indexing='ij')

        # 计算与中心点的距离
        d_squared = (x - ccol) ** 2 + (y - crow) ** 2

        # 计算高斯分布
        mask = torch.exp(-d_squared / (2 * (cutoff ** 2))).cuda()
        return mask
    # 高斯高通滤波器
    def gaussian_highpass_filter(size, cutoff):
        lowpass_mask = gaussian_lowpass_filter(size, cutoff).cuda()
        return 1 - lowpass_mask  # 高通滤波器是低通的反转

    # 对每个通道应用傅里叶变换，滤波，然后逆变换
    def apply_filter(image_tensor, filter_func, cutoff):
        image_tensor=image_tensor.to(dtype=torch.float32)
        filtered_image = torch.zeros_like(image_tensor).to(device=image_tensor.device)
        
        for c in range(channels):
            # 对每个通道进行傅里叶变换
            f_image = torch.fft.fft2(image_tensor[:, :, c])
            fshift = torch.fft.fftshift(f_image).to(device=image_tensor.device)  # 将低频移到中心
            
            # 创建高斯滤波器
            filter_mask = filter_func((rows, cols), cutoff).to(device=image_tensor.device)
            
            # 应用滤波器
            fshift_filtered = fshift * filter_mask
            
            # 逆傅里叶变换
            f_ishift = torch.fft.ifftshift(fshift_filtered)
            img_back = torch.fft.ifft2(f_ishift)
            img_back_magnitude = torch.abs(img_back)
            
            # 存储滤波后的图像
            filtered_image[:, :, c] = img_back_magnitude
            
        return filtered_image.to(dtype=torch.bfloat16)

    # 应用高斯低通滤波器
    lowpass_image_tensor = apply_filter(image, gaussian_lowpass_filter, cutoff_frequency)

    # 应用高斯高通滤波器
    highpass_image_tensor = apply_filter(image, gaussian_highpass_filter, cutoff_frequency)
    highpass_image_tensor=highpass_image_tensor.permute(2, 0, 1)
    lowpass_image_tensor=lowpass_image_tensor.permute(2, 0, 1)
    return highpass_image_tensor, lowpass_image_tensor

def setup_seeds():
    seed = 927

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser(description="CHAIR evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model")
parser.add_argument("--model_path", type=str, help="model_path")

parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
# TODO
parser.add_argument(
    "--data-path",
    type=str,
    default="/path/to/coco/val2014/",
    help="data path",
)
parser.add_argument("--batch-size", type=int, default=1)

parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--use-attn", action="store_true")
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--use-mask", action="store_true")
parser.add_argument("--use-cfg", action="store_true")
parser.add_argument("--gamma", type=float, default=2)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--max_new_tokens", type=int, default=512)

args = parser.parse_known_args()[0]

setup_seeds()

disable_torch_init()
tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, args.model)

# model_loader = ModelLoader(args.model, args.model_path)

base_dir = "playground/data/eval/chairs/" + args.model
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

coco_dataset = COCODataSet(data_path=args.data_path, trans=image_processor, model_config=model.config)
coco_loader = torch.utils.data.DataLoader(
    coco_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32
)

file_parts = [
    f"chair_eval",
    "_sample" if args.sample else "",
]

file_name = "".join(file_parts)

with open(os.path.join(base_dir, file_name + ".jsonl"), "w") as f:
    pass

for batch_id, data in tqdm(enumerate(coco_loader), total=len(coco_loader)):
    if batch_id == 500:
        break
    img_id = data["img_id"]
    image = data["image"]

    batch_size = img_id.shape[0]
    query = ["Please help me describe the image in detail."] * batch_size

    image_file = image[0]
    qs = query[0]
    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs


    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    


    input_ids = input_ids.to(device='cuda', non_blocking=True)
    image_tensor = image[0]
    highpass_image_tensor, lowpass_image_tensor = get_frequency_subimg(image_tensor.cuda())   
    # print(image_tensor.size()) 
    # highpass_image_tensor = highpass_image_tensor.unsqueeze(0)
    # lowpass_image_tensor = lowpass_image_tensor.unsqueeze(0)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),
            highpass_images=highpass_image_tensor.unsqueeze(0).half().cuda(),
            lowpass_images=lowpass_image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()

    # output_text = model_loader.decode(outputs)
    # print(len(outputs))

    with open(os.path.join(base_dir, file_name + ".jsonl"), "a") as f:
        json.dump({"image_id": int(img_id), "caption": outputs}, f)
        f.write("\n")
