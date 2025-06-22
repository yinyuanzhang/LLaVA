import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import faiss
from torchvision import models, transforms
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor
from transformers import CLIPImageProcessor, CLIPVisionModel


def build_faiss_index(embeddings, nlist=100):
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

    if embeddings.shape[0] > nlist * 2:
        index.train(embeddings)
    else:
        print("Using flat index due to small dataset.")
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    return index


def retrieve_top_k_with_faiss(query_emb, index, k=5):
    query_emb = np.array([query_emb])  # (1, D)
    D, I = index.search(query_emb, k)  # 返回距离和索引
    return I[0].tolist(), D[0].tolist()


def evaluate_retrieval_with_faiss(dataset_embeddings, ground_truth, image_ids, k=1, verbose=True):
    embeddings_np = np.array([e for e in dataset_embeddings])
    index = build_faiss_index(embeddings_np)

    correct = 0
    total = len(dataset_embeddings)

    for i in range(total):
        query_id = image_ids[i]
        query_emb = embeddings_np[i]

        _, indices = index.search(np.array([query_emb]), k+1)  # 包括自己
        retrieved_ids = [image_ids[idx] for idx in indices[0] if idx != i][:k]

        for img_id in retrieved_ids:
            if img_id in ground_truth.get(query_id, []):
                correct += 1
                break

    accuracy = correct / total
    if verbose:
        print(f"Top-{k} Accuracy: {accuracy:.4f}")
    return accuracy


def get_feature_extractor(model_name):
    if model_name == 'vgg':
        model = models.vgg16(pretrained=True)
        model = model.features # We only need the feature extraction part
    elif model_name == 'resnet':
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1]) # Remove the classification layer
    elif model_name == 'clip_base':
            clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            class CLIPFeatureExtractor(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def forward(self, x):
                    return self.model(x).pooler_output
            model = CLIPFeatureExtractor(clip_model)
    elif model_name == 'clip_large':
        clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
        class CLIPFeatureExtractor(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x).pooler_output
        model = CLIPFeatureExtractor(clip_model)
    elif model_name == 'mobilenet_v3':
        # 使用预训练的 MobileNetV3-Large 作为特征提取器
        mobilenet = models.mobilenet_v3_large(pretrained=True)
        # 只保留 feature extractor 部分
        feature_extractor = mobilenet.features
        model = feature_extractor


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

def get_default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_clip_large_transform():
    return transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

def get_mobilenet_v3_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def extract_features(image_path, model, device, model_type='resnet'):
    image = Image.open(image_path).convert('RGB')

    # 根据模型选择合适的预处理
    if model_type == 'clip_large':
        transform = get_clip_large_transform()  # 使用 336x336 分辨率
    elif model_type == 'mobilenet_v3':
        transform = get_mobilenet_v3_transform() # 224x224
    else:
        transform = get_default_transform()     # 使用 224x224 分辨率

    input_tensor = transform(image).unsqueeze(0) # Add batch dimension
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        features = model(input_tensor)
    
    if model_type in ['resnet', 'vgg']:
        features = features.view(features.size(0), -1)  # 展平 CNN 特征
    elif model_type in ['clip_large', 'mobilenet_v3']:
        features = features.view(features.size(0), -1)   # 取 [CLS] token 的特征


    features = F.normalize(features, p=2, dim=1).flatten() # Normalize
    
    return features.cpu().numpy()

if __name__ == "__main__":
    val2017_dir = "/users/zyy/autodl-tmp/playground/data/coco/val2017"
    dataset_images = {}
    N = 1000  # 可调整最大数量

    # 获取所有 .jpg 文件路径
    for img_file in os.listdir(val2017_dir):
        if img_file.endswith(".jpg"):
            image_id_str = os.path.splitext(img_file)[0]
            image_id = int(image_id_str)
            dataset_images[image_id] = os.path.join(val2017_dir, img_file)
            if len(dataset_images) >= N:
                break

    print(f"Total images: {len(dataset_images)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    vgg_model = get_feature_extractor('vgg').to(device)
    resnet_model = get_feature_extractor('resnet').to(device)
    clip_large = get_feature_extractor('clip_large').to(device)
    mobilenet_model = get_feature_extractor('mobilenet_v3').to(device)
    # yolo_backbone = get_yolo_backbone('yolov8s').to(device)

    # 提取特征
    dataset_features_vgg = []
    dataset_features_resnet = []
    dataset_features_yolo = []
    dataset_features_clip_large = []
    dataset_features_mobilenet = []

    for i, (image_id, img_path) in enumerate(dataset_images.items()):
        feat_vgg = extract_features(img_path, vgg_model, device)
        feat_resnet = extract_features(img_path, resnet_model, device)
        feat_clip_large = extract_features(img_path, clip_large, device, model_type='clip_large')
        feat_mobilenet = extract_features(img_path, mobilenet_model, device, model_type='mobilenet_v3')

        # feat_yolo = extract_yolo_global_feature(img_path, yolo_backbone, device)

        dataset_features_vgg.append(feat_vgg)
        dataset_features_resnet.append(feat_resnet)
        # dataset_features_yolo.append(feat_yolo)
        dataset_features_clip_large.append(feat_clip_large)
        dataset_features_mobilenet.append(feat_mobilenet)
        

    image_ids = list(dataset_images.keys())

    # 加载 ground truth
    annotations_file = "/users/zyy/autodl-tmp/playground/data/coco/annotations/instances_val2017.json"
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    image_id_to_categories = {}
    for annotation in data['annotations']:
        if annotation['image_id'] not in image_id_to_categories:
            image_id_to_categories[annotation['image_id']] = set()
        image_id_to_categories[annotation['image_id']].add(annotation['category_id'])

    ground_truth = {}
    for img_id in dataset_images.keys():
        current_cats = image_id_to_categories.get(img_id, set())
        gt_ids = [other_img_id for other_img_id, cats in image_id_to_categories.items()
                  if current_cats & cats and other_img_id != img_id]
        ground_truth[img_id] = gt_ids

    # 测试不同规模的数据集
    ks = [1,3]
    sizes = [100, 1000]

    for size in sizes:
        if size > len(dataset_images):
            continue
        print(f"\nTesting with {size} images...")

        for model_name, features in [
            ("VGG", dataset_features_vgg),
            ("ResNet", dataset_features_resnet),
            ("clip-large", dataset_features_clip_large),
            ("MobileNet-V3", dataset_features_mobilenet), 
        ]:
            print(f"  Evaluating {model_name} features...")
            embeddings = np.array(features[:size])
            ids = image_ids[:size]

            for k in ks:
                acc = evaluate_retrieval_with_faiss(embeddings, ground_truth, ids, k=k, verbose=False)
                print(f"    {model_name} Top-{k} Accuracy (size={size}): {acc:.4f}")

