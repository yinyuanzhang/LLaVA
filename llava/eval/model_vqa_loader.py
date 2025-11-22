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


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, yolo_model_path="./checkpoints/yolov/yolov8l-seg.pt"):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.yolo_model_path = yolo_model_path

        # Initialize YOLO model with configurable path
        self.yolo_model = YOLO(self.yolo_model_path).to('cpu')
        print(f"YOLO model loaded successfully: {self.yolo_model_path}")



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

        temp_image_folder = './image_folder'
        image_subfolder = os.path.join(temp_image_folder, os.path.splitext(image_file)[0])
        # if not os.path.exists(image_subfolder):
        #     os.makedirs(image_subfolder)

        src_image_path = os.path.join(self.image_folder, image_file)
        dest_image_path = os.path.join(image_subfolder, image_file)
        # if not os.path.exists(dest_image_path):
        #     shutil.copy(src_image_path, dest_image_path)

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
        #     # set window_size  336/14 = 24;   336/168 = 2;  336/112 = 3; 336/84 = 4; 
        #     window_size = 84

        #     masks = mask_tensor.float()
        #     mask_4d = masks.unsqueeze(1)
        #     pool = torch.nn.MaxPool2d(kernel_size=window_size, stride=window_size)
        #     window_mask = pool(mask_4d)
        #     window_mask = (window_mask.squeeze(1) > 0).int()
            
        #     num_patches_per_window = window_size // 1  # 每个窗口包含的小 patch 数量
        #     patch_mask = window_mask.repeat_interleave(num_patches_per_window, dim=1).repeat_interleave(num_patches_per_window, dim=2)[0, :, :]
         


        #     plt.figure(figsize=(10,5))
        #     plt.subplot(131)
        #     plt.imshow(image)
        #     plt.title('原始图像')

        #     plt.subplot(132)
        #     if len(corrected_mask.shape) == 3:
        #         plt.imshow(corrected_mask)
        #     else: 
        #         plt.imshow(corrected_mask, cmap='gray')  # 灰度显示
        #     plt.title('修正掩码')

        #     plt.subplot(133)  # 1 行 3 列，第 3 张
        #     plt.imshow(patch_mask)
        #     plt.title('第三张图片')

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
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, yolo_model_path="./checkpoints/yolov/yolov8l-seg.pt"):
    # mp.set_start_method('spawn')
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, yolo_model_path)
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


    questions_to_process = []
    if args.cache_mode == "write-only":
        processed_images = set()

    print("Filtering questions before data loader creation...")
    for line in tqdm(questions, desc="Pre-filtering questions"):

        if line.get('category') != 'random': # 使用 .get() 避免 KeyError，如果 'category' 不存在，则默认为 None
            continue # 跳过当前循环的其余部分，处理下一条数据

        # if line['question_id'] > 10000005:
        #     break

        if args.cache_mode == "write-only":
            current_image_filename = line['image']
    
            # 首先检查图片是否已经处理过，或者类别是否不是 'random'
            if current_image_filename in processed_images:
                continue # 如果图片已处理，则不将其添加到待处理列表中       
            processed_images.add(current_image_filename)
            questions_to_process.append(line)
        else:
            questions_to_process.append(line)

    print(f"Original questions count: {len(questions)}")
    print(f"Questions to process after pre-filtering: {len(questions_to_process)}")

    data_loader = create_data_loader(questions_to_process, args.image_folder, tokenizer, image_processor, model.config, yolo_model_path=args.yolo_model_path)

    for (input_ids, image_tensor, image_sizes, mask_tensor), line in tqdm(zip(data_loader, questions_to_process), total=len(questions_to_process)):
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
    

    try:
        # 打印缓存统计报告（适用于读取缓存的模式）
        if args.method_type in ["segmentation-cache", "fuzzy-cache", "cacheblend", "object-only"] and \
           args.cache_mode in ["read-only", "read-load"] and \
           hasattr(model.get_model(), 'stats'):
            print("--- Final Cache Statistics ---")
            # 使用新的统计报告方法（与Qwen2.5-VL一致）
            model.get_model().print_and_reset_stats()
    finally:
        # 确保缓存被正确保存
        if args.method_type in ["segmentation-cache", "fuzzy-cache"] and args.cache_mode == "write-only":
            if hasattr(model.get_model(), 'background_cache') and model.get_model().background_cache is not None:
                print("Evaluation finished. Saving segmentation/fuzzy cache...")
                model.get_model().background_cache.save()
                print("Background cache saved successfully.")
        elif args.method_type == "cacheblend" and args.cache_mode == "write-only":
            if hasattr(model.get_model(), 'kv_controller') and model.get_model().kv_controller is not None:
                print("Evaluation finished. Saving CacheBlend cache...")
                model.get_model().kv_controller.save_all()
                print("CacheBlend cache saved successfully.")

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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--method-type", type=str, default="native", choices=["native", "segmentation-cache", "object-only", "fuzzy-cache","cacheblend"])
    parser.add_argument("--cache-mode", type=str, default="read-only", choices=["read-only", "write-only", "read-load"])
    parser.add_argument("--dataset", type=str, default="default_dataset")
    # 【新增】轻量级query_key参数
    parser.add_argument("--use-lightweight-query-key", action="store_true", default=False, help="Use lightweight CNN backbone for query_key extraction instead of full ViT")
    parser.add_argument("--query-key-extractor-type", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "vgg11", "vgg13", "vgg16", "vgg19"], help="Type of lightweight query_key extractor")
    parser.add_argument("--similarity-threshold", type=float, default=0.1, help="Similarity threshold for cache matching (lower = stricter)")
    parser.add_argument("--yolo-model-path", type=str, default="./checkpoints/yolov/yolov8n-seg.pt", help="Path to YOLO segmentation model")
    parser.add_argument("--is-flexible-route", action="store_true", default=False, help="Enable flexible routing: use native encoding when cache misses")
    args = parser.parse_args()
    eval_model(args)
