import json
import hashlib

# 读取标准答案和预测结果文件，逐行解析为 JSON 对象
gt_data = [json.loads(line) for line in open('answer.jsonl', 'r', encoding='utf-8')]
pred_data = [json.loads(line) for line in open('submission.jsonl', 'r', encoding='utf-8')]

# 标签对应集合：GOLD* 表示标准答案，PRED* 表示预测结果
# 0: 完全匹配, 1: 部分匹配, 2: 不匹配
GOLD0, GOLD1, GOLD2 = set(), set(), set()
PRED0, PRED1, PRED2 = set(), set(), set()

# 处理标准答案：将每个候选项的哈希值按标签分类到不同集合中
for data in gt_data:
    for k, v in data['candidate'].items():
        # 确保没有重复键
        assert k not in GOLD0 and k not in GOLD1 and k not in GOLD2, f"Duplicate key found: {k}"
        if v == '完全匹配':
            GOLD0.add(k)
        elif v == '部分匹配':
            GOLD1.add(k)
        elif v == '不匹配':
            GOLD2.add(k)
        else:
            raise ValueError(f"Invalid label: {v}")

# 处理预测数据：计算每个候选项的哈希值并分类
for data in pred_data:
    for candidate in data['candidate']:
        # 用 text_id + text 计算唯一哈希值
        sha256_hash = hashlib.sha256((data['text_id'] + candidate['text']).encode('utf-8')).hexdigest()

        # 确保该哈希值在标准答案中出现过
        if sha256_hash not in GOLD0 and sha256_hash not in GOLD1 and sha256_hash not in GOLD2:
            raise ValueError(f"Hash {sha256_hash} not found in candidate keys")

        # 确保该哈希值没有重复记录
        assert sha256_hash not in PRED0 and sha256_hash not in PRED1 and sha256_hash not in PRED2, f"Duplicate key found: {sha256_hash}"

        # 分类存入对应标签集合
        if candidate['label'] == '完全匹配':
            PRED0.add(sha256_hash)
        elif candidate['label'] == '部分匹配':
            PRED1.add(sha256_hash)
        elif candidate['label'] == '不匹配':
            PRED2.add(sha256_hash)
        else:
            raise ValueError(f"Invalid label: {candidate['label']}")

# --- 计算指标 ---
# R: Recall, P: Precision, F1: F1 score
R0 = len(PRED0 & GOLD0) / len(GOLD0) if len(GOLD0) > 0 else 0.0
R1 = len(PRED1 & GOLD1) / len(GOLD1) if len(GOLD1) > 0 else 0.0
R2 = len(PRED2 & GOLD2) / len(GOLD2) if len(GOLD2) > 0 else 0.0

P0 = len(PRED0 & GOLD0) / len(PRED0) if len(PRED0) > 0 else 0.0
P1 = len(PRED1 & GOLD1) / len(PRED1) if len(PRED1) > 0 else 0.0
P2 = len(PRED2 & GOLD2) / len(PRED2) if len(PRED2) > 0 else 0.0

F10 = 2 * R0 * P0 / (R0 + P0) if (R0 + P0) > 0 else 0.0
F11 = 2 * R1 * P1 / (R1 + P1) if (R1 + P1) > 0 else 0.0
F12 = 2 * R2 * P2 / (R2 + P2) if (R2 + P2) > 0 else 0.0

#输出评估结果
#输出各类f1分数
print(f"完全匹配：{F10:.4f},部分匹配：{F11:.4f},不匹配：{F12:.4f}")
# Macro-F1：三类的平均值
macro_F1 = (F10 + F11 + F12) / 3.0
print(f"Macro-F1: {macro_F1:.4f}")
#计算加权平均
weighted_F1 = (R0 * F10 + R1 * F11 + R2 * F12) / (R0 + R1 + R2)
print(f"Weighted-F1: {weighted_F1:.4f}")
