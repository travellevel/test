import json

def filter_mechanics(input_file, output_file, keep_chinese_only=True):
    """
    从 JSONL 文件中筛选出 domain 为 'Mechanics' 的题目。

    参数:
        input_file (str): 输入 JSONL 文件路径（每行一个 JSON 对象）
        output_file (str): 输出文件路径
        keep_chinese_only (bool): 是否仅保留中文原题（剔除 translate: true 的英文翻译）
    """
    mechanics_problems = []
    total_count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_count += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ 跳过无法解析的行: {line[:50]}... 错误: {e}")
                continue

            # 检查 domain 字段是否为 Mechanics（忽略大小写及前后空格）
            domain = item.get("domain", "")
            if isinstance(domain, str) and domain.strip().lower() == "mechanics":
                # 根据选项决定是否过滤掉英文翻译题
                if keep_chinese_only:
                    # 保留 language == 'zh' 且 translate 不为 True 的条目
                    if item.get("language") == "zh" and not item.get("translate", False):
                        mechanics_problems.append(item)
                else:
                    mechanics_problems.append(item)

    # 将筛选结果写入新文件（每行一个 JSON 对象）
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in mechanics_problems:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ 处理完成！")
    print(f"   原始总条目: {total_count}")
    print(f"   筛选出力学题目: {len(mechanics_problems)}")
    print(f"   结果已保存至: {output_file}")

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 请根据实际文件路径修改以下两个变量
    input_path = "./test.jsonl"   # 原始数据文件
    output_path = "./mechanics_only.jsonl"    # 输出文件

    filter_mechanics(input_path, output_path, keep_chinese_only=True)