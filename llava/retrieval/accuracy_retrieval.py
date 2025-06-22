import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import functional as F
import os
import json
from ultralytics import YOLO
import torch.nn as nn

def get_feature_extractor(model_name):
    if model_name == 'vgg':
        model = models.vgg16(pretrained=True)
        model = model.features # We only need the feature extraction part
    elif model_name == 'resnet':
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1]) # Remove the classification layer
    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_large(pretrained=True)  # 也可以用 v3_small
        # 只保留到最后的 feature extractor，去掉分类层
        model = model.features    
    else:
        raise ValueError("Model name should be either 'vgg' or 'resnet'")
    
    model.eval() # Set model to evaluation mode
    return model

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path, model, device):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        features = model(input_tensor)

    # 对 MobileNetV3 进行全局平均池化 (GAP)，得到 [B, C]
    if isinstance(model, torch.nn.modules.container.Sequential) and \
       isinstance(model[-1], nn.modules.conv.Conv2d):  # 判断是否为 mobilenet_v3 的 features 输出
        gap = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        features = gap(features)

    features = features.view(features.size(0), -1) # Flatten
    features = F.normalize(features, p=2, dim=1) # Normalize
    
    return features.cpu().numpy()


def retrieve_top_k(query_feature, feature_list, top_k=5, exclude_id=None):
    similarities = []
    for img_id, feat in feature_list:
        if exclude_id is not None and img_id == exclude_id:
            continue  # 跳过自己
        sim = cosine_similarity(
            query_feature.reshape(1, -1),
            feat.reshape(1, -1)
        )[0][0]
        similarities.append(sim)

    indices = np.argsort(similarities)[::-1][:top_k]
    top_ids = [feature_list[i][0] for i in indices]
    top_scores = [similarities[i] for i in indices]
    return top_ids, top_scores

def evaluate_retrieval(dataset_features, ground_truth, k=1, verbose=True):
    correct = 0
    total = len(dataset_features)
    
    for idx, (query_img_id, query_feat) in enumerate(dataset_features):
        # 检索 Top-K（排除自己）
        top_ids, _ = retrieve_top_k(query_feat, dataset_features, top_k=k*2, exclude_id=query_img_id)

        # 判断前 k 个结果中是否有正确的匹配
        found = False
        for retrieved_img_id in top_ids[:k]:
            if retrieved_img_id in ground_truth.get(query_img_id, []):
                correct += 1
                found = True
                break
        
        # 可选：打印失败案例
        # if not found and verbose:
        #     print(f"Query ID {query_img_id} failed in Top-{k}")
            
    accuracy = correct / total
    if verbose:
        print(f"Top-{k} Accuracy: {accuracy:.4f}")
    return accuracy


def get_yolo_backbone(model_name='yolov8n'):
    model = YOLO(f'{model_name}.pt').model
    # 取出 backbone 和 neck 的前几层（不包括 detection head）
    backbone = model.model[:10]  # 根据具体模型调整层数
    backbone.eval()
    return backbone


if __name__ == "__main__":
    # COCO val2017 图像目录
    val2017_dir = "/users/zyy/autodl-tmp/playground/data/coco/val2017"
    dataset_images = {}
    N = 100

    # 获取所有 .jpg 文件路径
    for img_file in os.listdir(val2017_dir):
        if img_file.endswith(".jpg"):
            image_id_str = os.path.splitext(img_file)[0]
            image_id = int(image_id_str)  # 转换为整数ID
            dataset_images[image_id] = os.path.join(val2017_dir, img_file)
            if len(dataset_images) >= N:
                break

    print(f"Total images: {len(dataset_images)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    vgg_model = get_feature_extractor('vgg').to(device)
    resnet_model = get_feature_extractor('resnet').to(device)
    mobilenet_model = get_feature_extractor('mobilenet_v3').to(device)

    # yolo_backbone = get_yolo_backbone('yolov8n').to(device)

    dataset_features_vgg = []
    dataset_features_resnet = []
    dataset_features_mobilenet = []

    for i, (image_id, img_path) in enumerate(dataset_images.items()):
        print(f"Processing image {i+1}/{len(dataset_images)}: {img_path}")
        feat_vgg = extract_features(img_path, vgg_model, device)
        feat_resnet = extract_features(img_path, resnet_model, device)
        feat_mobilenet = extract_features(img_path, mobilenet_model, device)
        dataset_features_mobilenet.append((image_id, feat_mobilenet))
        dataset_features_vgg.append((image_id, feat_vgg))
        dataset_features_resnet.append((image_id, feat_resnet))

    # np.save("coco_val2017_vgg_features.npy", np.array([feat for _, feat in dataset_features_vgg]))
    # np.save("coco_val2017_resnet_features.npy", np.array([feat for _, feat in dataset_features_resnet]))


    annotations_file = "/users/zyy/autodl-tmp/playground/data/coco/annotations/instances_val2017.json"
    # 加载注解
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    # 构建 image_id 到 category_ids 的映射
    image_id_to_categories = {}
    for annotation in data['annotations']:
        if annotation['image_id'] not in image_id_to_categories:
            image_id_to_categories[annotation['image_id']] = set()
        image_id_to_categories[annotation['image_id']].add(annotation['category_id'])

    # 构建 ground truth：对于每个图像ID，找出属于相同类别的其他图像 IDs
    ground_truth = {}
    for img_id in dataset_images.keys():
        current_cats = image_id_to_categories.get(img_id, set())
        gt_ids = [other_img_id for other_img_id, cats in image_id_to_categories.items() 
                if current_cats & cats and other_img_id != img_id]
        ground_truth[img_id] = gt_ids


    ks = [1, 3, 5]
    accuracies = {}
    for k in ks:
        acc = evaluate_retrieval(dataset_features_vgg, ground_truth, k=k, verbose=False)
        accuracies[k] = acc

    # 打印结果
    for k, acc in accuracies.items():
        print(f"dataset_features_vgg Top-{k} Accuracy: {acc:.4f}")


    accuracies = {}
    for k in ks:
        acc = evaluate_retrieval(dataset_features_resnet, ground_truth, k=k, verbose=False)
        accuracies[k] = acc

    # 打印结果
    for k, acc in accuracies.items():
        print(f"dataset_features_resnet Top-{k} Accuracy: {acc:.4f}")


    accuracies = {}
    for k in ks:
        acc = evaluate_retrieval(dataset_features_mobilenet, ground_truth, k=k, verbose=False)
        accuracies[k] = acc

    # 打印结果
    for k, acc in accuracies.items():
        print(f"dataset_features_mobilenet Top-{k} Accuracy: {acc:.4f}")