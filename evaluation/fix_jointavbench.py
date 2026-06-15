# coding=gbk
import os
import json
import random
from typing import List, Tuple


# The following function is adapted from https://github.com/jointavbench/JointAVBench/blob/main/evaluation/evaluation.py
def generate_mcq_prompt(
    question: str,
    options: List[str],
    correct_answer: str,
    option_prefixes: List[str] = None
) -> Tuple[str, str]:
    """
    生成多选题的prompt文本并记录正确答案标记
    
    参数:
        question: 问题文本
        options: 选项文本列表
        correct_answer: 正确答案文本
        option_prefixes: 可选的选项前缀列表(如['A', 'B', 'C'])
        
    返回:
        tuple: (生成的prompt文本, 正确答案前缀如"A")
    """
    prefix2text = {}
    # 设置默认选项前缀(A., B., C., D.)
    if option_prefixes is None:
        option_prefixes = ['A', 'B', 'C', 'D']
    
    # 验证输入
    if len(options) < 2:
        raise ValueError("必须提供至少2个选项")
    if len(options) > len(option_prefixes):
        raise ValueError(f"选项数量({len(options)})超过前缀数量({len(option_prefixes)})")
    if correct_answer not in options:
        raise ValueError("正确答案不在提供的选项中")
    
    # 打乱选项顺序(但记录原始索引)
    indexed_options = list(enumerate(options))
    random.shuffle(indexed_options)
    
    # 构建带前缀的选项文本和正确答案跟踪
    prefixed_options = []
    correct_prefix = None
    
    for idx, (original_idx, option) in enumerate(indexed_options):
        prefix = option_prefixes[idx]
        prefixed_option = f"{prefix}. {option}"
        prefixed_options.append(prefixed_option)
        prefix2text[prefix] = option
        # 检查是否是正确答案
        if options[original_idx] == correct_answer:
            correct_prefix = prefix
    
    # 拼接问题和选项
    options_text = "\n".join(prefixed_options)
    prompt = f"{question}\n{options_text}"
    
    return prompt, correct_prefix, prefix2text


if __name__ == "__main__":
    jointavbench_dir = "JointAVBench"
    ori_qa_file = os.path.join(jointavbench_dir, "jointavbench.json")
    fixed_qa_file = os.path.join(jointavbench_dir, "jointavbench_fixed.json")

    random.seed(42)
    with open(ori_qa_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        prompt, correct_prefix, prefix2text = generate_mcq_prompt(item["question"], item["options"], item["correct_answer"])
        item["prompt"] = prompt
        item["correct_prefix"] = correct_prefix
        item["prefix2text"] = prefix2text
    with open(fixed_qa_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
