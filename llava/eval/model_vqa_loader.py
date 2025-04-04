import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, process_mask_images
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

import shutil
from ultralytics import YOLO
import torch.multiprocessing as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# class YOLOInference:
#     def __init__(self, model_path="yolov8l.pt"):
#         self.model = YOLO(model_path).eval()





# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

        # self.yolo_inference = YOLOInference(model_path="yolov8n-seg.pt")
        self.yolo_model = YOLO('yolov8n-seg.pt').to('cpu')
        # self.yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s").to('cpu')

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # os.path.join(self.image_folder, image_file) -> 复制到 os.path.join(temp_image_folder, image_file)
        temp_image_folder = './image_folder'
        image_subfolder = os.path.join(temp_image_folder, os.path.splitext(image_file)[0])
        if not os.path.exists(image_subfolder):
            os.makedirs(image_subfolder)

        src_image_path = os.path.join(self.image_folder, image_file)
        dest_image_path = os.path.join(image_subfolder, image_file)
        if not os.path.exists(dest_image_path):
            shutil.copy(src_image_path, dest_image_path)

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        # LLaVA 预处理流程：原图 → 居中填充为正方形 → 缩放到 336x336

        # 需要根据分割模型分理出背景和目标，生成mask。 1. llava对所有图像统一处理为 336*336，这里切割工具该怎么处理才能将mask映射回去
        result = self.yolo_model(src_image_path)

        # 计算缩放比例   
        orig_h, orig_w = result[0].orig_shape  # 原始尺寸（480,640）
        combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)


        if result[0].masks is not None:
            masks = result[0].masks.data.cpu().numpy().astype(np.uint8)   # masks尺寸（n, 480,640）
            
            for mask in masks:
                mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                combined_mask = np.bitwise_or(combined_mask, mask_resized)
        mask_pil = Image.fromarray(combined_mask)
        mask_tensor = process_mask_images([mask_pil], self.image_processor, self.model_config)[0]


        # mask图像还原
        image = image_tensor.numpy().transpose(1,2,0)  # (336,336,3)

        corrected_mask = np.transpose(mask_tensor, (1, 2, 0))[:, :, 0] # 新形状 (336,336,3)
        # 若为单通道灰度图
        if corrected_mask.shape[-1] == 1:
            corrected_mask = np.squeeze(corrected_mask, axis=-1)

        # if True:
        #     plt.figure(figsize=(10,5))
        #     plt.subplot(121)
        #     plt.imshow(image)
        #     plt.title('原始图像')

        #     plt.subplot(122)
        #     if len(corrected_mask.shape) == 3:
        #         plt.imshow(corrected_mask)
        #     else: 
        #         plt.imshow(corrected_mask, cmap='gray')  # 灰度显示
        #     plt.title('修正掩码')
        #     # plt.show()
        #     plt.savefig(dest_image_path)


        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size, mask_tensor

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, mask_tensors = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    mask_tensors = torch.stack(mask_tensors, dim=0)
    return input_ids, image_tensors, image_sizes, mask_tensors


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    mp.set_start_method('spawn', force=True)
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, model_args = args)
    model.eval()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes, mask_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                masks=mask_tensor.to(dtype=torch.uint8, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
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
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--image-cache", type=bool, default=True)
    args = parser.parse_args()

    eval_model(args)
