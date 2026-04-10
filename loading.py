from datasets import load_dataset

# 加载数据集
ds = load_dataset("Eureka-Lab/PHYBench")

def filter_mechanics(example):
    """筛选 tag 包含 MECHANICS 的样本"""
    tag = example.get("tag")
    # 若 tag 是字符串
    if isinstance(tag, str):
        return tag == "MECHANICS"
    # 若 tag 是列表（多标签）
    elif isinstance(tag, list):
        return "MECHANICS" in tag
    return False

# 对每个 split 应用筛选
ds_mechanics = ds.filter(filter_mechanics)

# 保存筛选后的数据为 JSON Lines
for split_name, dataset in ds_mechanics.items():
    dataset.to_json(f"./phybench_mechanics_{split_name}.jsonl")

print("保存完成，各 split 样本数：", {k: len(v) for k, v in ds_mechanics.items()})