import pandas as pd
import numpy as np
import requests
from sklearn.cluster import KMeans
import os
from dotenv import load_dotenv
load_dotenv()

# 1. 配置Jina API
url = 'https://api.jina.ai/v1/embeddings'
headers = {
    'Content-Type': 'application/json',
    'Authorization': f"Bearer {os.getenv("JINA_API_KEY")}"
}

# 2. 读取游戏描述数据
df = pd.read_csv("90-文档-Data/灭神纪/游戏描述.csv")
texts = df['description'].tolist() # Shape of df['description'] is (10,)

# 3. 获取文本嵌入
data = {
    "model": "jina-embeddings-v3",
    "task": "text-matching",
    "dimensions": 1024,
    "normalized": True,
    "input": texts
}

response = requests.post(url, headers=headers, json=data)

if response.status_code != 200:
    raise RuntimeError(f"API调用失败: {response.status_code} - {response.text}")

embeddings = [item['embedding'] for item in response.json().get('data', [])]
if not embeddings:
    raise RuntimeError("API未返回嵌入向量")

embeddings = np.array(embeddings) # Shape of embeddings is (n, 1024)

# 4. 聚类分析
# Note: 对embedding vectors进行无监督聚类分析，把语义上相似的文本自动分组。
# n_clusters是K-Means算法的重要参数，即把数据分成$(n_clusters)个簇(clusters)。
# random_state参数用于保证分类结果的可复现性，因为K-Means算法的初始质心是随机
# 选择的，这会导致每次运行结果可能会有不同，这里显示指定random_state可以保证随机
# 初始化的方式都是一样的。42本身无特殊含义，只要是非负整数即可。
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(embeddings)

# 5. 打印结果
print("\n聚类结果：")
for i, lbl in enumerate(labels):
    print(f"Cluster {lbl}: {texts[i]}")
