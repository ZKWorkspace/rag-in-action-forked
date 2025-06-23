from collections import Counter
import math
# 猢狲的战斗日志
battle_logs = [
    "猢狲，施展，烈焰拳，击退，妖怪；随后开启，金刚体，抵挡，神兵，攻击。",
    "妖怪，使用，寒冰箭，攻击，猢狲，但被，烈焰拳，反击，击溃。",
    "猢狲，召唤，烈焰拳，与，毁灭咆哮，击败，妖怪，随后，收集，妖怪，精华。"
]
# 超参数
k1 = 1.5
b = 0.75
# 构建词表
vocabulary = set(word for log in battle_logs for word in log.split("，"))
vocab_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
idx_to_vocab = {idx: word for word, idx in vocab_to_idx.items()}
# 计算IDF (逆文档频率：Inverse Document Frequency)
N = len(battle_logs)
# Note: 统计每个词在多少个文档中出现过，一个文档中的重复词只计一次
df = Counter(word for log in battle_logs for word in set(log.split("，")))
# Note: 词在越多文档中出现，IDF越低，说明这个词约没有区分度
idf = {word: math.log((N - df[word] + 0.5) / (df[word] + 0.5) + 1) for word in vocabulary}
# 日志长度信息
avg_log_len = sum(len(log.split("，")) for log in battle_logs) / N
# BM25稀疏嵌入生成
def bm25_sparse_embedding(log):
    tf = Counter(log.split("，"))
    log_len = len(log.split("，"))
    embedding = {}
    for word, freq in tf.items():
        if word in vocabulary:
            idx = vocab_to_idx[word]
            score = idf[word] * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * log_len / avg_log_len))
            embedding[idx] = score
    return embedding
# 生成稀疏向量
for log in battle_logs:
    sparse_embedding = bm25_sparse_embedding(log)
# print(f"稀疏嵌入： {sparse_embedding}")
    for idx in sparse_embedding.keys():
        word = idx_to_vocab[idx]
        print(f"稀疏嵌入：{word}: {sparse_embedding[idx]}")
    print("-" * 100)