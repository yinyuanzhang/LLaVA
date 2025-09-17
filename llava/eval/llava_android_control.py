import os
import sys
import json
import argparse
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Any
import torch
import warnings
import shortuuid
import traceback
from PIL import Image
import cv2

# LLaVA 相关导入
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, process_mask_images

# YOLO 相关导入
from ultralytics import YOLO

# 导入评估脚本
try:
    from evaluate_android_control import evaluate_android_control_action, evaluate_type_only
except ImportError as e:
    print(f"Error importing evaluation module: {e}")
    print("请确保 'evaluate_android_control.py' 在当前目录中")
    sys.exit(1)


def generate_mask_for_image(image_path: str, yolo_model_path: str = "./checkpoints/yolov/yolov8n-seg.pt"):
    """
    为图像生成mask，使用YOLO模型进行目标检测和分割
    
    Args:
        image_path (str): 图像路径
        yolo_model_path (str): YOLO模型路径
        
    Returns:
        PIL.Image: 生成的mask图像
    """
    try:
        # 初始化YOLO模型
        yolo_model = YOLO(yolo_model_path)
        
        # 运行检测
        result = yolo_model(image_path)
        
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


def set_seed(seed: int):
    """设置随机种子以确保可复现性。"""
    print(f"Setting random seed to {seed}...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_jobs_from_filtered_data(image_root: str, eval_file: str, eval_type: str, thinking: bool, use_simplified_prompt: bool = False) -> List[Dict]:
    """
    根据过滤后的数据集目录结构生成任务列表。
    适配 LLaVA 模型调用，但保持 Qwen2.5-VL 的数据处理逻辑。
    """
    with open(eval_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    # 将原始数据转换为字典，方便根据 episode_id 和 step 快速查找
    original_data_map = {}
    for entry in original_data:
        episode_id = str(entry['episode']['episode_id'])
        original_data_map[episode_id] = entry

    jobs = []
    
    for subdir_name in tqdm(os.listdir(image_root), desc='Generating jobs from filtered data'):
        # 解析子目录名称以获取 episode_id
        parts = subdir_name.split('_')
        if not parts or not parts[0].isdigit():
            continue
        episode_id = parts[0]
        
        # 确保原始数据中存在该 episode_id
        if episode_id not in original_data_map:
            warnings.warn(f"Warning: Episode ID {episode_id} not found in original eval file. Skipping.")
            continue
            
        original_entry = original_data_map[episode_id]
        subdir_path = os.path.join(image_root, subdir_name)
        
        # 获取子目录中的所有图片
        image_files = [f for f in os.listdir(subdir_path) if f.endswith('.png')]
        
        for img_file in image_files:
            # 解析图片名称以获取 step_id
            try:
                step = int(img_file.split('.')[0].split('_')[1])
            except (IndexError, ValueError):
                warnings.warn(f"Warning: Could not parse step from image file {img_file}. Skipping.")
                continue

            # 从原始数据中获取对应的 step_pam 和 step_check_pam
            if step >= len(original_entry["step_check_pams"]):
                warnings.warn(f"Warning: Step {step} out of bounds for episode {episode_id}. Skipping.")
                continue

            step_check_pam = original_entry["step_check_pams"][step]
            step_pam = original_entry["step_pams"][step]
            step_instruction = original_entry["episode"]["step_instructions"][step]
            
            width, height = original_entry['width'], original_entry['height']
            # LLaVA 使用固定分辨率，这里使用默认值
            h_bar, w_bar = 336, 336  
            
            system_prompt = "You are a helpful assistant."
            if thinking:
                system_prompt = "The screen is cut and reassembled with the background in front and the foreground in the back. You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags.\nDuring the reasoning process, identify and state the sub-goal of the current step by enclosing it within <step> </step> tags."
            
            # 根据是否使用简化版prompt生成不同的tool_system_prompt
            if use_simplified_prompt:
                tool_system_prompt = "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\\n{\\\"type\\\": \\\"function\\\", \\\"function\\\": {\\\"name\\\": \\\"mobile_use\\\", \\\"description\\\": \\\"Use a touchscreen to interact with a mobile device, and take screenshots.\\\\n* This is an interface to a mobile device with touchscreen. You can perform actions like swiping, typing, clicking, etc.\\\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\\\n* The screen's resolution is " + str(w_bar) + "x" + str(h_bar) + ".\\\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\\\", \\\"parameters\\\": {\\\"properties\\\": {\\\"action\\\": {\\\"description\\\": \\\"The action to perform. The available actions are:\\\\n* `key`: Perform a key event on the mobile device.\\\\n* `click`: Click the point on the screen.\\\\n* `long_press`: Press the point on the screen.\\\\n* `swipe`: Swipe from one point to another.\\\\n* `type`: Input text into the activated input box.\\\\n* `system_button`: Press the system button.\\\\n* `open`: Open an app on the device.\\\\n* `wait`: Wait for the change to happen.\\\\n* `terminate`: Terminate the current task.\\\", \\\"enum\\\": [\\\"key\\\", \\\"click\\\", \\\"long_press\\\", \\\"swipe\\\", \\\"type\\\", \\\"system_button\\\", \\\"open\\\", \\\"wait\\\", \\\"terminate\\\"], \\\"type\\\": \\\"string\\\"}}, \\\"required\\\": [\\\"action\\\"], \\\"type\\\": \\\"object\\\"}}}}\\n</tools>\\n\\nPlease carefully consider what action to perform. For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": \\\"mobile_use\\\", \\\"arguments\\\": {\\\"action\\\": \\\"<action_type>\\\"}}\\n</tool_call>"
            else:
                tool_system_prompt = "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\\n{\\\"type\\\": \\\"function\\\", \\\"function\\\": {\\\"name\\\": \\\"mobile_use\\\", \\\"description\\\": \\\"Use a touchscreen to interact with a mobile device, and take screenshots.\\\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\\\n* The screen's resolution is " + str(w_bar) + "x" + str(h_bar) + ".\\\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\\\", \\\"parameters\\\": {\\\"properties\\\": {\\\"action\\\": {\\\"description\\\": \\\"The action to perform. The available actions are:\\\\n* `key`: Perform a key event on the mobile device.\\\\n    - This supports adb's `keyevent` syntax.\\\\n    - Examples: \\\\\\\"volume_up\\\\\\\", \\\\\\\"volume_down\\\\\\\", \\\\\\\"power\\\\\\\", \\\\\\\"camera\\\\\\\", \\\\\\\"clear\\\\\\\".\\\\n* `click`: Click the point on the screen with coordinate (x, y).\\\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\\\n* `type`: Input the specified text into the activated input box.\\\\n* `system_button`: Press the system button.\\\\n* `open`: Open an app on the device.\\\\n* `wait`: Wait specified seconds for the change to happen.\\\\n* `terminate`: Terminate the current task and report its completion status.\\\", \\\"enum\\\": [\\\"key\\\", \\\"click\\\", \\\"long_press\\\", \\\"swipe\\\", \\\"type\\\", \\\"system_button\\\", \\\"open\\\", \\\"wait\\\", \\\"terminate\\\"], \\\"type\\\": \\\"string\\\"}, \\\"coordinate\\\": {\\\"description\\\": \\\"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.\\\", \\\"type\\\": \\\"array\\\"}, \\\"coordinate2\\\": {\\\"description\\\": \\\"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.\\\", \\\"type\\\": \\\"array\\\"}, \\\"text\\\": {\\\"description\\\": \\\"Required only by `action=key`, `action=type`, and `action=open`.\\\", \\\"type\\\": \\\"string\\\"}, \\\"time\\\": {\\\"description\\\": \\\"The seconds to wait. Required only by `action=long_press` and `action=wait`.\\\", \\\"type\\\": \\\"number\\\"}, \\\"button\\\": {\\\"description\\\": \\\"Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`\\\", \\\"enum\\\": [\\\"Back\\\", \\\"Home\\\", \\\"Menu\\\", \\\"Enter\\\"], \\\"type\\\": \\\"string\\\"}, \\\"status\\\": {\\\"description\\\": \\\"The status of the task. Required only by `action=terminate`.\\\", \\\"type\\\": \\\"string\\\", \\\"enum\\\": [\\\"success\\\", \\\"failure\\\"]}}, \\\"required\\\": [\\\"action\\\"], \\\"type\\\": \\\"object\\\"}}}}\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call>"

            system_message = system_prompt + tool_system_prompt

            # 构建 task_progress
            task_progress = []
            for n in range(step):
                if n < len(original_entry["step_pams"]):
                    task_progress.append(json.dumps(original_entry["step_pams"][n], ensure_ascii=False))

            if eval_type == 'low':
                user_message_template = "The user query:  {goal}\\nCurrent step query: {step_instruction}\\nTask progress (You have done the following operation on the current device): {task_progress}"
                user_message = user_message_template.format(
                    goal=original_entry["episode"]["goal"], 
                    step_instruction=step_instruction, 
                    task_progress=''.join([f'Step {n+1}: {tp}; ' for n, tp in enumerate(task_progress)])
                )
            elif eval_type == 'high':
                user_message_template = "The user query:  {goal}\\nTask progress (You have done the following operation on the current device): {task_progress}"
                user_message = user_message_template.format(
                    goal=original_entry["episode"]["goal"], 
                    task_progress=''.join([f'Step {n+1}: {tp}; ' for n, tp in enumerate(task_progress)])
                )
            else:
                raise ValueError(f"Invalid eval type: {eval_type}")
            
            # 创建 job 字典
            job_dict = {
                'question_id': f"{episode_id}_{step}",
                'episode_id': episode_id,
                'image': os.path.join(subdir_name, img_file),
                'text': user_message,  # LLaVA 使用 'text' 字段
                'system_message': system_message,
                'width': width,
                'height': height,
                'resized_width': w_bar,
                'resized_height': h_bar,
                'check_pams': step_check_pam,
                'use_simplified_prompt': use_simplified_prompt,
            }
            jobs.append(job_dict)

    return jobs


def parse_model_output(output: str, thinking: bool) -> str:
    """解析模型输出，提取工具调用JSON。"""
    pred = output
    if thinking and '</think>' in pred:
        pred = pred.split('</think>')[-1]
    
    if '<tool_call>' in pred:
        pred = pred.split('<tool_call>')[1]
    else:
        # Fallback for outputs that don't use <tool_call> tags properly
        if '{"name": "mobile_use", "arguments":' in pred:
            pred = '{"name": "mobile_use", "arguments":' + pred.split('{"name": "mobile_use", "arguments":', 1)[1]
        else:
            return None # 无法解析，返回None
            
    if '</tool_call>' in pred:
        pred = pred.split('</tool_call>')[0]
    else:
        pred = pred.split("<conclusion>")[0]
        # Another fallback
        if '}}' in pred:
            pred = pred.rsplit('}}', 1)[0] + '}}'
        else:
            return None # 无法解析，返回None

    return pred.strip()


def parse_simplified_output(output: str, thinking: bool) -> str:
    """解析简化版prompt的输出，提取动作类型。"""
    pred = output.strip()
    if thinking and '</think>' in pred:
        pred = pred.split('</think>')[-1].strip()
    
    if '<tool_call>' in pred and '</tool_call>' in pred:
        # 提取tool_call标签内容
        tool_call_content = pred.split('<tool_call>')[1].split('</tool_call>')[0]
        try:
            parsed = json.loads(tool_call_content)
            if 'arguments' in parsed and 'action' in parsed['arguments']:
                action = parsed['arguments']['action']
                # 验证动作类型是否有效
                valid_actions = ['wait', 'system_button', 'type', 'open', 'swipe', 'long_press', 'click']
                if action in valid_actions:
                    return action
        except:
            pass
    
    return None


class LLaVAModel:
    """LLaVA模型包装器，模拟Qwen2VLChat的接口"""
    
    def __init__(self, model_path, model_base, temperature=0.0, conv_mode="vicuna_v1", 
                 method_type="native", cache_mode="read-only", dataset="android_control",
                 yolo_model_path="./checkpoints/yolov/yolov8n-seg.pt", args=None):
        self.temperature = temperature
        self.conv_mode = conv_mode
        self.method_type = method_type
        self.cache_mode = cache_mode
        self.dataset = dataset
        self.yolo_model_path = yolo_model_path
        
        # 初始化 LLaVA 模型
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        # 关键修复：传递 model_args 参数
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name, model_args=args
        )
        self.model.eval()
    
    def generate(self, job_dict):
        """生成响应"""
        image_file = job_dict['image']
        # 构建完整的query，包含system message和user message
        system_message = job_dict.get('system_message', '')
        user_message = job_dict['text']

        # 构建完整的prompt，类似Qwen的full_prompt
        if system_message:
            full_prompt = system_message + "\\n\\n" + user_message
        else:
            full_prompt = user_message
            
        # 添加图片占位符，直接使用完整prompt作为最终输入
        if self.model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\\n' + full_prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\\n' + full_prompt

        # 处理图像
        image = Image.open(image_file).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        
        # 处理masks（如果使用segmentation相关方法）
        masks = None
        if self.method_type in ["segmentation-cache", "object-only"]:
            # 生成mask
            mask_pil = generate_mask_for_image(image_file, self.yolo_model_path)
            # 对mask进行与图像相同的预处理
            mask_tensor = process_mask_images([mask_pil], self.image_processor, self.model.config)[0]
            # 修复数据类型：使用 uint8 而非 half
            masks = mask_tensor.unsqueeze(0).to(dtype=torch.uint8, device='cuda', non_blocking=True)
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.float16, device='cuda', non_blocking=True),
                masks=masks,
                image_sizes=[image.size],
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=None,
                num_beams=1,
                max_new_tokens=10,
                use_cache=True
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs


def run_evaluation(args):
    """
    主评估函数，整合了数据加载、推理和评分。
    """
    set_seed(args.seed)

    print("--- Step 1: Generating AndroidControl jobs ---")
    jobs = generate_jobs_from_filtered_data(
        eval_file=args.eval_file,
        image_root=args.image_root,
        eval_type=args.eval_type,
        thinking=args.thinking,
        use_simplified_prompt=args.use_simplified_prompt
    )

    print("--- Step 2: Loading LLaVA model ---")
    model = LLaVAModel(
        model_path=args.model_path,
        model_base=args.model_base,
        temperature=args.temperature,
        conv_mode=args.conv_mode,
        method_type=args.method_type,
        cache_mode=args.cache_mode,
        dataset=args.dataset,
        yolo_model_path=args.yolo_model_path,
        args=args  # 关键修复：传递完整的 args
    )

    print("--- Step 3: Running inference ---")
    ans_file = open(args.answers_file, "w")
    
    for job in tqdm(jobs, desc=f"Processing AndroidControl jobs"):
        try:
            # 准备输入给 LLaVA 的字典
            llava_input_dict = {
                'image': os.path.join(args.image_root, job['image']),
                'text': job['text'],
                'system_message': job.get('system_message', '')
            }

            # 调用 generate
            response = model.generate(llava_input_dict)
            
            job['llm_output'] = response
            ans_file.write(json.dumps(job) + '\\n')
            ans_file.flush()

        except Exception as e:
            warnings.warn(f"Error during inference for job {job.get('question_id', 'N/A')}: {e}")
            traceback.print_exc()
            job['llm_output'] = f"Error: {e}"
            ans_file.write(json.dumps(job) + '\\n')
            continue

    ans_file.close()
    print(f"\\nInference completed. Raw results saved to {args.answers_file}")
    
    print("--- Step 4: Computing scores ---")
    if args.use_simplified_prompt:
        compute_scores_simplified(args, jobs)
    else:
        compute_scores(args, jobs)

    try:
        # 可以在这里执行一些收尾工作，比如打印统计报告
        if hasattr(model.model, 'get_model') and hasattr(model.model.get_model(), 'stats_collector') and model.model.get_model().stats_collector:
            print("--- Final Cache Statistics ---")
            model.model.get_model().stats_collector.report_stats(args.dataset)
    finally:
        # 关键修复：添加缓存清理逻辑，与 model_vqa_loader.py 保持一致
        if args.method_type in ["segmentation-cache", "fuzzy-cache"] and args.cache_mode == "write-only":
            if hasattr(model.model, 'get_model') and hasattr(model.model.get_model(), 'background_cache'):
                model.model.get_model().background_cache.close()
                print("Cache closed after write-only phase.")
        elif args.method_type in ["segmentation-cache", "fuzzy-cache"] and args.cache_mode in ["read-only", "read-load"]:
            if hasattr(model.model, 'get_model') and hasattr(model.model.get_model(), 'stats_collector'):
                model.model.get_model().stats_collector.report_stats(args.dataset)


def compute_scores_simplified(args, jobs: List[Dict]):
    """计算和打印评估分数 - 专门用于简化版prompt。"""
    type_match_num = 0
    error_num = 0
    action_type_stats = {}  # 统计各动作类型的准确率
    
    print("Using simplified prompt evaluation mode")
    
    for job in tqdm(jobs, desc='Computing scores (simplified)'):
        output = job.get('llm_output')
        if not output or 'Error' in output:
            error_num += 1
            continue

        current_check_pam = job['check_pams']
        gt_action = current_check_pam['action']
        
        # 初始化统计
        if gt_action not in action_type_stats:
            action_type_stats[gt_action] = {'total': 0, 'correct': 0}
        action_type_stats[gt_action]['total'] += 1

        try:
            pred_action_type = parse_simplified_output(output, thinking=args.thinking)
            if pred_action_type is None:
                error_num += 1
                continue
            
            # 构造简化的pred_action用于记录
            pred_action = {'action': pred_action_type}
            job['llm_prediction'] = pred_action
            
            # 只进行类型匹配评估
            type_match = evaluate_type_only(pred_action, current_check_pam)
            if type_match:
                type_match_num += 1
                action_type_stats[gt_action]['correct'] += 1
                job['type_match'] = True

        except Exception as e:
            warnings.warn(f"Error evaluating job {job.get('question_id', 'N/A')}: {e}")
            traceback.print_exc()
            error_num += 1
            continue
    
    # 打印结果
    print(f'Type_match_num: {type_match_num} / {len(jobs)} = {type_match_num/len(jobs)*100:.2f}%')
    print(f'Error num: {error_num}')
    
    # 打印各动作类型准确率
    print("\\n=== Action Type Statistics ===")
    for action, stats in action_type_stats.items():
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"{action}: {stats['correct']}/{stats['total']} = {acc:.2f}%")
    
    res = {
        'total_jobs': len(jobs),
        'type_match_acc': type_match_num/len(jobs)*100 if len(jobs) > 0 else 0,
        'error_num': error_num,
        'evaluation_mode': 'simplified',
        'action_type_stats': action_type_stats
    }
    
    print(f"\\n=== Summary ===")
    print(json.dumps({k: v for k, v in res.items() if k != 'action_type_stats'}, indent=2))
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, args.model_path.split('/')[-1], 'android_control', args.eval_type)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f'scores_simplified_{timestamp}.json'), 'w') as f:
        json.dump(res, f, indent=2)
    
    # 保存精简版jobs
    jobs_to_save = []
    for job in jobs:
        job_copy = {
            'question_id': job.get('question_id'),
            'image_path': job.get('image'),
            'llm_output': job.get('llm_output'),
            'llm_prediction': job.get('llm_prediction'),
            'ground_truth_action': job.get('check_pams', {}).get('action'),
            'type_match': job.get('type_match', False),
            'use_simplified_prompt': True,
        }
        jobs_to_save.append(job_copy)

    with open(os.path.join(output_dir, f'jobs_simplified_{timestamp}.json'), 'w') as f:
        json.dump(jobs_to_save, f, indent=2)


def compute_scores(args, jobs: List[Dict]):
    """计算和打印评估分数。"""
    Type_match_num = 0
    Extact_match_num = 0
    click_match_num = 0
    all_click_num = 0
    error_num = 0
    
    for job in tqdm(jobs, desc='Computing scores'):
        output = job.get('llm_output')
        if not output or 'Error' in output:
            error_num += 1
            continue

        current_check_pam = job['check_pams']

        try:
            pred_json_str = parse_model_output(output, thinking=False)
            if pred_json_str is None:
                error_num += 1
                continue
                
            pred_action = json.loads(pred_json_str)

            job['llm_prediction'] = pred_action
            
            # 使用新的类型匹配评估函数
            type_match = evaluate_type_only(pred_action, current_check_pam)
            
            # 保留原有的精确匹配评估
            _, extact_match = evaluate_android_control_action(
                pred_action, current_check_pam, 
                job['width'], job['height'], 
                job['resized_width'], job['resized_height'], 
                pred_type='abs_resized', gt_type='original_resized'
            )

            if type_match:
                Type_match_num += 1
                job['type_match'] = True
            
            if extact_match:
                Extact_match_num += 1
                job['extact_match'] = True
                
            if extact_match and pred_action['action'] == 'click':
                click_match_num += 1
                
            if current_check_pam['action'] == 'click':
                all_click_num += 1

        except Exception as e:
            warnings.warn(f"Error evaluating job {job.get('question_id', 'N/A')}: {e}")
            traceback.print_exc()
            error_num += 1
            continue
            
    print('Type_match_num and Extact_match_num: ', Type_match_num, Extact_match_num, f'/ all = {len(jobs)}')
    print('click_match_num:', click_match_num, f'/ all = {all_click_num}')
    print('error num', error_num)

    res = {
        'total_jobs': len(jobs),
        'type_match_acc': Type_match_num/len(jobs)*100 if len(jobs) > 0 else 0,
        'extact_match_acc': Extact_match_num/len(jobs)*100 if len(jobs) > 0 else 0,
        'click_match_acc': click_match_num/all_click_num*100 if all_click_num > 0 else 0,
        'error_num': error_num,
    }
    
    print(json.dumps(res, indent=' '))

    # 保存结果
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, args.model_path.split('/')[-1], 'android_control', args.eval_type)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f'scores_{timestamp}.json'), 'w') as f:
        json.dump(res, f, indent=2)
    
    # 保存精简版jobs
    jobs_to_save = []
    for job in jobs:
        job_copy = {
            'question_id': job.get('question_id'),
            'image_path': job.get('image'),
            'llm_output': job.get('llm_output'),
            'llm_prediction': job.get('llm_prediction'),
            'ground_truth': job.get('check_pams'),
            'type_match': job.get('type_match', False),
            'extact_match': job.get('extact_match', False)
        }
        jobs_to_save.append(job_copy)

    with open(os.path.join(output_dir, f'jobs_{timestamp}.json'), 'w') as f:
        json.dump(jobs_to_save, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA AndroidControl Evaluation Inference Script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the LLaVA model.")
    parser.add_argument("--model-base", type=str, default=None, help="Path to the LLaVA model base.")
    parser.add_argument("--eval-file", type=str, required=True, help="Path to the evaluation file.")
    parser.add_argument("--image-root", type=str, required=True, help="Path to the image root directory.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--eval-type", type=str, required=True, choices=['high', 'low'], help="Evaluation type.")
    parser.add_argument("--thinking", action='store_true', help="Enable thinking mode.")
    parser.add_argument("--use-simplified-prompt", action='store_true', default=True)
    parser.add_argument("--answers-file", type=str, default="android_control_answers.jsonl", help="Output JSONL file for results.")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1", help="Conversation mode.")
    
    # LLaVA 特定的参数
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results.")
    parser.add_argument("--dataset", type=str, default="android_control")
    parser.add_argument('--cache-mode', type=str, default='read-only', help='Cache mode for inference.')
    
    parser.add_argument("--method-type", type=str, default="native", 
                       choices=["native", "segmentation-cache", "object-only", "fuzzy-cache"],
                       help="Method type: native (original), segmentation-cache (bg/fg cache), object-only (fg only), fuzzy-cache (whole image cache)")
    parser.add_argument("--yolo-model-path", type=str, default="./checkpoints/yolov/yolov8n-seg.pt",
                       help="Path to YOLO model for mask generation")
    
    args = parser.parse_args()
    print(args)
    run_evaluation(args)