import argparse
import time
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

def get_frequency_subimg(image, cutoff_frequency = 30):
    # 高斯低通滤波器
    image=image.permute(1, 2, 0)
    rows, cols, channels= image.shape
    assert channels == 3
    def gaussian_lowpass_filter(size, cutoff):
        rows, cols = size
        crow, ccol = rows // 2, cols // 2
        mask = torch.zeros((rows, cols), dtype=torch.float32).cuda()
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

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            image_tensor = process_images([image], image_processor, model.config)[0]
            
            highpass_image_tensor, lowpass_image_tensor = get_frequency_subimg(image_tensor.cuda())
            # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    highpass_images=highpass_image_tensor.unsqueeze(0).half().cuda(),
                    lowpass_images=lowpass_image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    eval_model(args)
