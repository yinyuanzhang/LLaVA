import json
from pathlib import Path

# 设置路径
data_dir = Path("/users/zyy/autodl-tmp/playground/data/coco")
annotations_path = data_dir / "annotations" / "instances_val2017.json"

# 加载注解
with open(annotations_path, 'r') as f:
    data = json.load(f)

# 构建 image_id 到 category_ids 的映射
image_to_categories = {}
for annotation in data['annotations']:
    if annotation['image_id'] not in image_to_categories:
        image_to_categories[annotation['image_id']] = set()
    image_to_categories[annotation['image_id']].add(annotation['category_id'])


def evaluate_retrieval(query_image_id, retrieved_image_ids, image_to_categories):
    query_categories = image_to_categories[query_image_id]
    relevant_retrievals = sum([1 for img_id in retrieved_image_ids if not image_to_categories[img_id].isdisjoint(query_categories)])
    return relevant_retrievals / len(retrieved_image_ids)


# 应该用不到
# 如果需要将 category_id 转换为具体的类别名称，可以使用 categories 字段
category_id_to_name = {category['id']: category['name'] for category in data['categories']}