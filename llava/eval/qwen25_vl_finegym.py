import os
import sys
import json
import argparse
import re
import shortuuid
from tqdm import tqdm
from typing import List, Dict, Any
import warnings
import traceback
import math
from PIL import Image

# Assume these are available from your Qwen2.5-VL repository setup
try:
    from qwen2_vl.model import Qwen2VLChat
    
    def dump_image(line_dict, img_root):
        """
        Processes a single image for VQA evaluation and returns its full path.
        The 'image' key in line_dict is a relative path to the image file.
        """
        image_path = os.path.join(img_root, line_dict['image'])
        
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        return image_path
    
except ImportError as e:
    print(f"Error importing Qwen2.5-VL specific modules: {e}")
    sys.exit(1)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
            
            question_text = f"The image shows one frame from a sequence of a gymnastics move. Gymnastics moves are divided into the following four categories: Vault-Women, Floor Exercise, Balance Beam, and Uneven Bars. Please carefully distinguish the differences between these actions and identify which category is shown in the image. Return only the category name, without any explanation. Vault-Women: The gymnast runs, jumps off a springboard, performs a mid-air acrobatic move over a vaulting horse, and lands. Floor Exercise: The gymnast performs acrobatic tumbling passes, jumps, and dance moves on a padded floor. Balance Beam: The gymnast performs a routine of acrobatic skills, dance elements, and turns on a narrow beam. Uneven Bars: The gymnast performs swinging, rotational, and release moves between two different height bars."
            # question_text = f"The image shows a gymnast performing a move. Please identify the gymnastics event from the following four categories:\n\n**Vault-Women** (The gymnast uses a springboard to perform a series of acrobatic moves, such as flips and twists, above a vaulting horse.)\n**Floor Exercise** (The gymnast performs a series of tumbling passes, jumps, and dance moves on a padded floor.)\n**Balance Beam** (The gymnast performs a series of balancing skills, flips, leaps, and turns on a narrow beam.)\n**Uneven Bars** (The gymnast performs a series of swinging, rotational, and release moves between two different height bars.)\n\nPlease carefully distinguish the differences between these actions based on the relationship between the gymnast and the equipment, and identify which category is shown in the image.\n\nReturn only the category name, without any other text."
#             question_text = f"""Focus on the gymnastics equipment visible in this image:

# **Vault-Women**: Springboard and vaulting table apparatus
# **Floor Exercise**: Large padded floor area (no elevated equipment)
# **Balance Beam**: Narrow elevated beam (4 inches wide)
# **Uneven Bars**: Two horizontal bars at different heights

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

def run_finegym_inference(args):
    """
    Main function to run inference on FineGym using Qwen2.5-VL.
    """
    questions = load_finegym_annotations(
        args.val_element_txt,
        args.categories_txt,
        args.image_folder,
        args.n_frames
    )
    
    def get_chunk(lst, n, k):
        chunk_size = math.ceil(len(lst) / n)
        return lst[k*chunk_size:(k+1)*chunk_size]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)

    def qwen_dump_image_wrapper(line_dict_from_model):
        return dump_image(line_dict_from_model, args.image_folder)

    cot_prompt = ""
    if args.use_cot:
        cot_prompt = args.cot_prompt if args.cot_prompt else " If you are uncertain..., Determine whether to think step by step..."

    print(f"Loading Qwen2.5-VL model from {args.model_path}")
    
    # 根据method_type确定默认cache_mode
    if args.method_type in ["segmentation-cache", "fuzzy-cache", "cacheblend"]:
        default_cache_mode = "read-load"
    else:
        default_cache_mode = args.cache_mode
        
    model = Qwen2VLChat(
        model_path=args.model_path,
        temperature=args.temperature,
        use_custom_prompt=True,
        cache_mode=default_cache_mode,
        dataset=args.dataset,
        method_type=args.method_type,
        use_image_segmentation=args.use_image_segmentation,
        # 【新增】传递轻量级query_key参数
        use_lightweight_query_key=args.use_lightweight_query_key,
        query_key_extractor_type=args.query_key_extractor_type,
        similarity_threshold=args.similarity_threshold
    )
    model.set_dump_image(qwen_dump_image_wrapper)

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
            
            # 确定当前样本的cache_mode
            current_cache_mode = default_cache_mode
            if args.method_type in ["segmentation-cache", "fuzzy-cache", "cacheblend"]:
                # 提取帧信息，判断是否是第一帧
                frame_info = question_id.split('_')[-1]  # 获取帧号，如 "1", "2", etc.
                if frame_info == "1":  # 第一帧使用write-only
                    current_cache_mode = "write-only"
                    print(f"First frame detected for {question_id}, using cache_mode: {current_cache_mode}")
                else:  # 后续帧使用read-load
                    current_cache_mode = "read-load"
                    
            # 设置当前样本的cache_mode
            if hasattr(model, 'set_temp_cache_mode'):
                model.set_temp_cache_mode(current_cache_mode)
            elif hasattr(model.model, 'cache_mode'):
                # 直接修改模型内部的cache_mode
                model.model.cache_mode = current_cache_mode
            
            qwen_input_dict = {
                'image': image_relative_path,
                'question': question_text,
                'task': 'finegym',
                'index': question_id
            }

            messages = model.build_prompt(qwen_input_dict, dataset=args.dataset)

            if args.use_cot and len(messages) > 0 and messages[-1]['type'] == 'text':
                messages[-1]['value'] += cot_prompt

            response = model.generate(messages)

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
        "segmentation_enabled": args.use_image_segmentation
    }

    all_results.append({"summary": final_results})

    # 一次性将所有结果写入文件
    with open(args.answers_file, "w") as f:
        # 使用 json.dump 来写入整个列表，保证文件格式一致
        json.dump(all_results, f, indent=4)


    print(f"\nInference completed. Results saved to {args.answers_file}")
    print(f"Total instances evaluated: {total_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Final Accuracy: {accuracy:.2f}%")

    # 与 POPE 保持一致的缓存统计与持久化逻辑（包括 CacheBlend）
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-VL FineGym Evaluation Script")

    parser.add_argument("--model-path", type=str, required=True, help="Path to the Qwen2.5-VL model")
    parser.add_argument("--image-folder", type=str, required=True, help="Root folder for extracted frames.")
    parser.add_argument("--answers-file", type=str, default="finegym_vqa_answers.jsonl", help="Output file path for generated answers.")
    
    parser.add_argument("--val-element-txt", type=str, required=True, help="Path to the gym99_val_element.txt file.")
    parser.add_argument("--categories-txt", type=str, required=True, help="Path to the gym99_categories.txt file.")
    parser.add_argument("--set-categories-txt", type=str, required=True, help="Path to the set_categories.txt file.")
    parser.add_argument("--n-frames", type=int, default=5, help="Number of frames extracted per instance.")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--use-cot", action="store_true")
    parser.add_argument("--cot-prompt", type=str, default="")
    parser.add_argument("--cache-mode", type=str, default="read-only")
    parser.add_argument("--dataset", type=str, default="finegym")
    parser.add_argument("--annotation-json", type=str, default="finegym")    

    parser.add_argument("--use-image-segmentation", action="store_true", default=False)
    parser.add_argument("--yolo-model-path", type=str, default="./checkpoints/yolov/yolov8l-seg.pt")
    
    parser.add_argument("--method-type", type=str, default="native",
                       choices=["native", "segmentation-cache", "object-only", "fuzzy-cache", "cacheblend"],
                       help="Method type: native (original), segmentation-cache (bg/fg cache), object-only (fg only), fuzzy-cache (whole image cache), cacheblend (selective recomputation with cached image patches)")

    # 【新增】轻量级query_key提取器参数
    parser.add_argument("--use-lightweight-query-key", action="store_true", default=True,
                       help="Enable lightweight query_key extractor (default: True)")
    parser.add_argument("--query-key-extractor-type", type=str, default="resnet18",
                       choices=["resnet18", "resnet34", "resnet50", "resnet101", "vgg11", "vgg13", "vgg16", "vgg19"],
                       help="Type of lightweight query_key extractor (default: resnet18)")

    # 【新增】相似度阈值参数
    parser.add_argument("--similarity-threshold", type=float, default=0.5,
                       help="Similarity threshold for cache matching (default: 0.5)")

    args = parser.parse_args()
    
    run_finegym_inference(args)
