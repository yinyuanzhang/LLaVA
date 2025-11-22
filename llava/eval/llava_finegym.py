import os
import json
import argparse
import re
from tqdm import tqdm
import warnings
import traceback
import math
import numpy as np
import torch
import cv2
from PIL import Image

# LLaVA 相关导入
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, process_mask_images

# YOLO 相关导入（用于mask生成）
try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: YOLO not available. Segmentation features will be disabled.")
    YOLO = None

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def set_seed(seed: int):
    """设置随机种子以确保可复现性。"""
    print(f"Setting random seed to {seed}...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_finegym_annotations(val_element_txt, categories_txt, image_root_dir, n_frames=5):
    """
    Loads and processes FineGym annotations, generating one VQA question for each frame,
    with event-level categories for the prompt.
    """
    print("Loading FineGym annotations and categories...")

    # Define the 4 event-level categories directly
    event_categories = {
        '1': 'Floor Exercise',
        '2': 'Balance Beam',
        '3': 'Uneven Bars',
        '4': 'Vault-Women'
    }
    event_labels_prompt = ", ".join(event_categories.values())

    # 1. Load element-to-set mapping from gym99_categories.txt
    element_to_set_map = {}
    with open(categories_txt, 'r') as f:
        for line in f:
            match = re.match(r'Clabel:\s*(?P<clabel>\d+);\s*set:\s*(?P<set_id>\d+);.*', line.strip())
            if match:
                data = match.groupdict()
                element_to_set_map[data['clabel']] = data['set_id']

    # 2. Process the validation element file
    questions = []
    try:
        with open(val_element_txt, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Validation file not found at {val_element_txt}.")

    # 3. Load the set-to-event mapping
    # This mapping is derived from the set_categories.txt provided earlier
    set_to_event_map = {
        '21': 'Floor Exercise', '22': 'Floor Exercise', '23': 'Floor Exercise', '24': 'Floor Exercise', '25': 'Floor Exercise',
        '31': 'Balance Beam', '32': 'Balance Beam', '33': 'Balance Beam', '34': 'Balance Beam', '35': 'Balance Beam',
        '41': 'Uneven Bars', '42': 'Uneven Bars', '43': 'Uneven Bars', '44': 'Uneven Bars',
        '1': 'Vault-Women',
    }

    # 4. Process the validation data and generate one question per frame
    for line in tqdm(lines, desc="Generating prompts"):
        match = re.match(r'(?P<full_instance_id>\S+)\s(?P<element_id>\d+)', line.strip())
        if match:
            data = match.groupdict()
            full_instance_id = data['full_instance_id']
            element_id = data['element_id']
            video_id = full_instance_id.split('_')[0]

            set_id = element_to_set_map.get(element_id)
            if not set_id:
                continue

            # Map from set_id to event_name
            ground_truth_event = set_to_event_map.get(set_id)
            if not ground_truth_event:
                continue

            # question_text = """
            # [ ROLE ]
            # You are an expert gymnastics judge.

            # [ TASK ]
            # Your task is to analyze the given image, strictly adhere to the definitions below, and identify which women's gymnastics category it belongs to.

            # [ CATEGORY DEFINITIONS & KEY VISUAL CUES ]

            # # * **Uneven Bars:** **Two bars set at different heights**.
            # # * **Vault-Women:** **Vaulting horse** (a large stationary block) and/or **springboard**.
            # # * **Floor Exercise:** **Large, padded square floor**.
            # # * **Balance Beam:** **Single, narrow, raised beam**.

            # [ OUTPUT REQUIREMENT ]
            # Return **only** the English category name you have identified.

            # Example:
            # `Uneven Bars`
            # """



            # question_text = """
            # [ ROLE ]
            # You are an expert gymnastics judge.

            # [ TASK ]
            # Your task is to analyze the given image, strictly adhere to the definitions below, and identify which women's gymnastics category it belongs to.

            # [ CATEGORY EQUIPMENT ]

            # # * **Uneven Bars:** **Bar**.
            # # * **Vault-Women:** **Vaulting horse**.
            # # * **Floor Exercise:** **No elevated equipments**.
            # # * **Balance Beam:** **Balance Beam**.

            # [ OUTPUT REQUIREMENT ]
            # Return **only** the English category name you have identified.

            # Example:
            # `Uneven Bars`
            # """


            question_text = """
There are four categories of women's gymnastics:
Uneven Bars: Horizontal bars.
Balance Beam: A single, narrow, elevated beam.
Vault-Women: A springboard and a vaulting table (a support platform).
Floor Exercise: No other elevated apparatus.
The image shows one frame from a sequence of a gymnastics move. Based on the image, please analyze which category of gymnastics this move belongs to. Return only the category name.
            """




            # question_text = f"The image shows one frame from a sequence of a gymnastics move. This move belongs to **exactly one** of the following four **distinct and mutually exclusive** categories: Vault-Women, Floor Exercise, Balance Beam, and Uneven Bars. Please describe the content of the image. and then return the category name this image belonged to. Please ensure the category name matches the action shown in the image."

            # question_text = f"The image shows one frame from a sequence of a gymnastics move. Gymnastics moves are divided into the following four different categories: Vault-Women, Floor Exercise, Balance Beam, and Uneven Bars. Return only the category name from this image, without any explanation."



            # question_text = """The image shows one frame from a sequence of a gymnastics move. Gymnastics moves are divided into the following four categories: Vault-Women, Floor Exercise, Balance Beam, and Uneven Bars. Please carefully distinguish the differences between these actions and identify which category is shown in the image. Return only the category name, without any explanation.
            # """
            
            # question_text = """
            # The image is a single frame of a gymnast. Your task is to identify which one of the following four distinct categories the gymnastics move belongs to.

            # **Your decision must be based solely on the apparatus definitions below.**

            # **Categories & Apparatus:**

            # * **Vault-Women:** **Vaulting horse** (a large stationary block) and/or **springboard**.
            # * **Floor Exercise:** **Large, padded square floor**.
            # * **Balance Beam:** **Single, narrow, raised beam**.
            # * **Uneven Bars:** **Two bars set at different heights**.

            # Return only the category name, and explanation.
            # """


            # question_text = """请描述这张图片。"""

            ### prompt3 
#             question_text = f"""Focus on the gymnastics equipment visible in this image:
# **Uneven Bars**: Women:Horizontal bars.
# **Vault-Women**: Women:Springboard and vaulting table apparatus
# **Floor Exercise**: Women:Large padded floor area (no elevated equipment)
# **Balance Beam**: Women:Narrow elevated beam (4 inches wide)
# Identify the event based on the equipment. Return only the category name."""
            


            # question_text = """You will be shown a single frame of a gymnast. Your task is to identify which **one** of the following **four distinct categories** the gymnastics move belongs to, based on the definitions provided.

            # **Categories & Definitions:**

            # * **Vault-Women:** The gymnast interacts with a **springboard** and a **vaulting horse** (a large stationary block). The action typically involves running towards it or being in mid-air over it.
            # * **Floor Exercise:** The gymnast performs on a **large, padded square floor**. The image will likely show tumbling, jumping, or dance poses on this floor.
            # * **Balance Beam:** The gymnast performs on a **single, narrow, raised beam**. All action takes place on or directly above this beam.
            # * **Uneven Bars:** The gymnast performs on or between **two bars set at different heights**. The action involves swinging, rotating, or releasing from these bars.

            # Return only the category name, and the reason why it belongs to."""


            # **Categories & Definitions:**

            # * **Vault-Women:** The gymnast interacts with a **springboard** and a **vaulting horse** (a large stationary block). The action typically involves running towards it or being in mid-air over it.
            # * **Floor Exercise:** The gymnast performs on a **large, padded square floor**. The image will likely show tumbling, jumping, or dance poses on this floor.
            # * **Balance Beam:** The gymnast performs on a **single, narrow, raised beam**. All action takes place on or directly above this beam.
            # * **Uneven Bars:** The gymnast performs on or between **two bars set at different heights**. The action involves swinging, rotating, or releasing from these bars.


            ## prompt1
            # question_text = f"The image shows one frame from a sequence of a gymnastics move. Gymnastics moves are divided into the following four categories: Vault-Women, Floor Exercise, Balance Beam, and Uneven Bars. Please carefully distinguish the differences between these actions and identify which category is shown in the image. Return only the category name, without any explanation. Vault-Women: The gymnast runs, jumps off a springboard, performs a mid-air acrobatic move over a vaulting horse, and lands. Floor Exercise: The gymnast performs acrobatic tumbling passes, jumps, and dance moves on a padded floor. Balance Beam: The gymnast performs a routine of acrobatic skills, dance elements, and turns on a narrow beam. Uneven Bars: The gymnast performs swinging, rotational, and release moves between two different height bars."

            # question_text = f"The image shows one frame from a sequence of a gymnastics move. Gymnastics moves are divided into the following four categories: Vault-Women, Floor Exercise, Balance Beam, and Uneven Bars. Please carefully distinguish the differences between these actions and identify which category is shown in the image. Vault-Women: The gymnast runs, jumps off a springboard, performs a mid-air acrobatic move over a vaulting horse, and lands. Floor Exercise: The gymnast performs acrobatic tumbling passes, jumps, and dance moves on a padded floor. Balance Beam: The gymnast performs a routine of acrobatic skills, dance elements, and turns on a narrow beam. Uneven Bars: The gymnast performs swinging, rotational, and release moves between two different height bars. Return only the category name, without any explanation."

            ### prompt2          
            # question_text = f"The image shows a gymnast performing a move. Please identify the gymnastics event from the following four categories:\n\n**Vault-Women** (The gymnast uses a springboard to perform a series of acrobatic moves, such as flips and twists, above a vaulting horse.)\n**Floor Exercise** (The gymnast performs a series of tumbling passes, jumps, and dance moves on a padded floor.)\n**Balance Beam** (The gymnast performs a series of balancing skills, flips, leaps, and turns on a narrow beam.)\n**Uneven Bars** (The gymnast performs a series of swinging, rotational, and release moves between two different height bars.)\n\nPlease carefully distinguish the differences between these actions based on the relationship between the gymnast and the equipment, and identify which category is shown in the image.\n\nReturn only the category name, without any other text."

#             ### prompt3 
#             question_text = f"""Focus on the gymnastics equipment visible in this image:
# **Vault-Women**: Springboard and vaulting table apparatus
# **Floor Exercise**: Large padded floor area (no elevated equipment)
# **Balance Beam**: Narrow elevated beam (4 inches wide)
# **Uneven Bars**: Two horizontal bars at different heights.
# Identify the event based on the equipment. Return only the category name."""

            # Generate prompts for EACH frame in the instance
            for i in range(1, n_frames + 1):
                image_dir = os.path.join(image_root_dir, video_id, full_instance_id)
                image_file_name = f"frame_{i:03d}.jpg"
                image_relative_path = os.path.join(video_id, full_instance_id, image_file_name)

                full_image_path = os.path.join(image_root_dir, image_relative_path)
                if not os.path.isfile(full_image_path):
                    continue

                questions.append({
                    "question_id": f"{full_instance_id}_{i}",
                    "image": image_relative_path,
                    "text": question_text,
                    "category": ground_truth_event,  # The ground truth event name
                    "choices": list(event_categories.values())
                })

        if len(questions) >= 1000:
            questions = questions[0:500]
            break

    print(f"Loaded {len(questions)} VQA questions from FineGym.")
    return questions

class LLaVAModel:
    """LLaVA模型包装器，模拟Qwen2VLChat的接口用于FineGym评估"""

    def __init__(self, model_path, model_base, temperature=0.0, conv_mode="vicuna_v1",
                 method_type="native", cache_mode="read-only", dataset="finegym",
                 yolo_model_path="./checkpoints/yolov/yolov8n-seg.pt", args=None):
        self.temperature = temperature
        self.conv_mode = conv_mode
        self.method_type = method_type
        self.cache_mode = cache_mode
        self.dataset = dataset
        self.yolo_model_path = yolo_model_path

        # 初始化 YOLO 模型（只在需要时初始化）
        self.yolo_model = None
        if self.method_type in ["segmentation-cache", "object-only", "cacheblend"]:
            if YOLO is not None:
                print(f"Initializing YOLO model from: {self.yolo_model_path}")
                self.yolo_model = YOLO(self.yolo_model_path).to('cpu')
                print(f"YOLO model loaded successfully: {self.yolo_model_path}")
            else:
                print("Warning: YOLO not available. Segmentation features will be disabled.")

        # 初始化 LLaVA 模型
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name, model_args=args
        )
        self.model.eval()

    def generate_mask_for_image(self, image_path: str):
        """
        为图像生成mask，使用已初始化的YOLO模型
        """
        try:
            # 运行检测
            result = self.yolo_model(image_path)

            # 获取原始图像尺寸
            orig_h, orig_w = result[0].orig_shape
            combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

            # 如果检测到目标，合并所有mask
            if result[0].masks is not None:
                masks = result[0].masks.data.cpu().numpy().astype(np.uint8)

                for mask in masks:
                    # 确保mask尺寸正确
                    if mask.shape != (orig_h, orig_w):
                        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    combined_mask = np.bitwise_or(combined_mask, mask)

            # 转换为PIL图像
            mask_pil = Image.fromarray(combined_mask)
            return mask_pil

        except Exception as e:
            warnings.warn(f"Error generating mask for {image_path}: {e}")
            # 返回空的mask作为fallback
            image = Image.open(image_path)
            w, h = image.size
            return Image.fromarray(np.zeros((h, w), dtype=np.uint8))

    def generate(self, job_dict):
        """生成响应，兼容FineGym数据格式"""
        image_file = job_dict['image']
        user_message = job_dict['text']

        # image_file = "/data/zyy/autodl-tmp/playground/data/eval/pope/val2014/COCO_val2014_000000000073.jpg"
        # user_message = "请描述这张图片。"
        # user_message = 'Is there a snowboard in the image?\nAnswer the question using a single word or phrase.'
        # 构建prompt，为FineGym任务使用简单的user prompt
        if self.model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + user_message
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + user_message

        from llava.conversation import conv_templates
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 处理图像
        image = Image.open(image_file).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        # 处理masks（如果使用segmentation相关方法）
        masks = None
        if self.method_type in ["segmentation-cache", "object-only", "cacheblend"]:
            # 使用实例方法生成mask
            mask_pil = self.generate_mask_for_image(image_file)
            # 对mask进行与图像相同的预处理
            mask_tensor = process_mask_images([mask_pil], self.image_processor, self.model.config)[0]
            # 修复数据类型：使用 uint8 而非 half
            masks = mask_tensor.unsqueeze(0).to(dtype=torch.uint8, device='cuda', non_blocking=True)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        # shape = (1, 3, 336, 336)
        # masks = torch.zeros(shape)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),
                masks=masks,
                image_sizes=[image.size],
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)
        return outputs

def run_finegym_inference(args):
    """
    主函数运行FineGym推理
    """
    set_seed(args.seed)

    questions = load_finegym_annotations(
        args.val_element_txt,
        args.categories_txt,
        args.image_folder,
        args.n_frames
    )

    # 过滤
    # questions = questions[0:30]

    def get_chunk(lst, n, k):
        chunk_size = math.ceil(len(lst) / n)
        return lst[k*chunk_size:(k+1)*chunk_size]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)

    cot_prompt = ""
    args.use_cot = False
    if args.use_cot:
        cot_prompt = args.cot_prompt if args.cot_prompt else " Please think step by step to identify the gymnastics event category."

    print(f"Loading LLaVA model from {args.model_path}")

    # 根据method_type确定默认cache_mode
    if args.method_type in ["segmentation-cache", "fuzzy-cache", "cacheblend"]:
        default_cache_mode = "read-load"
    else:
        default_cache_mode = args.cache_mode

    model = LLaVAModel(
        model_path=args.model_path,
        model_base=args.model_base,
        temperature=args.temperature,
        conv_mode=args.conv_mode,
        method_type=args.method_type,
        cache_mode=default_cache_mode,
        dataset=args.dataset,
        yolo_model_path=args.yolo_model_path,
        args=args
    )

    ans_file = open(args.answers_file, "w")
    correct_count = 0
    total_count = 0
    all_results = []

    for i, q_line in tqdm(enumerate(questions), total=len(questions), desc="Running inference"):
        try:
            image_relative_path = q_line["image"]
            question_text = q_line["text"]
            question_id = q_line["question_id"]
            ground_truth_event_name = q_line['category']

            # 为COT添加提示
            if args.use_cot:
                question_text += cot_prompt

            # 准备输入给 LLaVA 的字典
            llava_input_dict = {
                'image': os.path.join(args.image_folder, image_relative_path),
                'text': question_text
            }

            # 对齐 Qwen 行为：按帧切换缓存模式
            # 第一帧使用 write-only 写入缓存，后续帧使用 read-load 复用缓存
            try:
                current_cache_mode = default_cache_mode
                if args.method_type in ["segmentation-cache", "fuzzy-cache", "cacheblend"]:
                    frame_info = str(question_id).split('_')[-1]
                    if frame_info == "1":
                        current_cache_mode = "write-only"
                    else:
                        current_cache_mode = "read-load"

                    # 安全设置到模型（同时设置 wrapper 与内部模型，确保各分支读取一致）
                    if hasattr(model, 'model'):
                        # 顶层包装器（LlavaLlamaForCausalLM）
                        if hasattr(model.model, 'cache_mode'):
                            model.model.cache_mode = current_cache_mode
                        # 内部架构模型（LlavaLlamaModel / CacheBlendLlavaLlamaModel）
                        if hasattr(model.model, 'get_model') and callable(getattr(model.model, 'get_model')):
                            inner = model.model.get_model()
                            if hasattr(inner, 'cache_mode'):
                                inner.cache_mode = current_cache_mode
                        # CacheBlend 需要同步内部状态（参考 qwen 实现）
                        if args.method_type == "cacheblend" and hasattr(model.model, '_update_cacheblend_state'):
                            try:
                                model.model._update_cacheblend_state()
                            except Exception:
                                pass
            except Exception:
                # 不因缓存模式设置失败而中断评估
                pass

            # 调用 generate
            response = model.generate(llava_input_dict)

            # 评估时，检查返回的文本中是否包含正确的event名称
            model_is_correct = ground_truth_event_name in response
            if model_is_correct:
                correct_count += 1
            total_count += 1

            result_entry = {
                "question_id": question_id,
                "prompt": question_text,
                "text": response,
                "ground_truth_event": ground_truth_event_name,
                "is_correct": model_is_correct
            }
            all_results.append(result_entry)

        except Exception as e:
            warnings.warn(f"Error processing question ID {q_line.get('question_id', 'N/A')}. Error: {e}")
            traceback.print_exc()
            continue

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

    final_results = {
        "final_accuracy": f"{accuracy:.2f}%",
        "correct_predictions": correct_count,
        "total_instances_evaluated": total_count,
        "model_name": args.model_path,
        "method_type": args.method_type,
        "use_image_segmentation": args.use_image_segmentation if hasattr(args, 'use_image_segmentation') else False
    }

    all_results.append({"summary": final_results})

    # 一次性将所有结果写入文件
    with open(args.answers_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nInference completed. Results saved to {args.answers_file}")
    print(f"Total instances evaluated: {total_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Final Accuracy: {accuracy:.2f}%")

    try:
        # 打印缓存统计报告（适用于读取缓存的模式）
        if args.method_type in ["segmentation-cache", "fuzzy-cache", "cacheblend", "object-only"] and \
           args.cache_mode in ["read-only", "read-load"] and \
           hasattr(model.model.model, 'stats'):
            print("--- Final Cache Statistics ---")
            # 使用新的统计报告方法（与Qwen2.5-VL一致）
            model.model.model.print_and_reset_stats()
    finally:
        # 确保缓存被正确保存
        if args.method_type in ["segmentation-cache", "fuzzy-cache"] and args.cache_mode == "write-only":
            if hasattr(model.model.model, 'background_cache') and model.model.model.background_cache is not None:
                print("Evaluation finished. Saving segmentation/fuzzy cache...")
                model.model.model.background_cache.save()
                print("Background cache saved successfully.")
        elif args.method_type == "cacheblend" and args.cache_mode == "write-only":
            if hasattr(model.model.model, 'kv_controller') and model.model.model.kv_controller is not None:
                print("Evaluation finished. Saving CacheBlend cache...")
                model.model.model.kv_controller.save_all()
                print("CacheBlend cache saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA FineGym Evaluation Script")

    parser.add_argument("--model-path", type=str, required=True, help="Path to the LLaVA model")
    parser.add_argument("--model-base", type=str, default=None, help="Path to the LLaVA model base.")
    parser.add_argument("--image-folder", type=str, required=True, help="Root folder for extracted frames.")
    parser.add_argument("--answers-file", type=str, default="finegym_vqa_answers.json", help="Output file path for generated answers.")

    parser.add_argument("--val-element-txt", type=str, required=True, help="Path to the gym99_val_element.txt file.")
    parser.add_argument("--categories-txt", type=str, required=True, help="Path to the gym99_categories.txt file.")
    parser.add_argument("--set-categories-txt", type=str, required=True, help="Path to the set_categories.txt file.")
    parser.add_argument("--n-frames", type=int, default=5, help="Number of frames extracted per instance.")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results.")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--use-cot", action="store_true")
    parser.add_argument("--cot-prompt", type=str, default="")
    parser.add_argument("--cache-mode", type=str, default="read-only")
    parser.add_argument("--dataset", type=str, default="finegym")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1", help="Conversation mode.")

    # LLaVA 特定参数（与android_control保持一致）
    parser.add_argument("--use-image-segmentation", action="store_true", default=False)
    parser.add_argument("--yolo-model-path", type=str, default="./checkpoints/yolov/yolov8n-seg.pt")

    parser.add_argument("--method-type", type=str, default="native",
                       choices=["native", "segmentation-cache", "object-only", "fuzzy-cache", "cacheblend"],
                       help="Method type: native (original), segmentation-cache (bg/fg cache), object-only (fg only), fuzzy-cache (whole image cache), cacheblend (selective recomputation with cached image patches)")

    # 轻量级 query_key 相关新参数（与 model_vqa_loader.py 保持一致）
    parser.add_argument("--use-lightweight-query-key", action="store_true", default=False,
                       help="Use lightweight CNN backbone for query_key extraction instead of full ViT")
    parser.add_argument("--query-key-extractor-type", type=str, default="resnet18",
                       choices=["resnet18", "resnet34", "resnet50", "resnet101", "vgg11", "vgg13", "vgg16", "vgg19"],
                       help="Type of lightweight query_key extractor")
    parser.add_argument("--similarity-threshold", type=float, default=0.1,
                       help="Similarity threshold for cache matching (lower = stricter)")

    args = parser.parse_args()
    print(args)
    run_finegym_inference(args)
