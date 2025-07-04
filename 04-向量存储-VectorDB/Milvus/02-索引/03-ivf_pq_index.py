from pymilvus import MilvusClient, DataType
import random
import time
import numpy as np

# 1. 设置 Milvus 客户端
import os
from dotenv import load_dotenv

load_dotenv()
client = MilvusClient(uri="http://172.17.19.130:19530", token=os.getenv("MILVUS_TOKEN"))
COLLECTION_NAME = "coll_03_ivf_pq_index"

# 如果集合已存在，则删除
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

# 2. 创建 schema
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)

# 3. 创建集合
client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

# 4. 插入随机向量数据
num_vectors = 1024
vectors = [[random.random() for _ in range(128)] for _ in range(num_vectors)]
ids = list(range(num_vectors))
entities = [{"id": ids[i], "vector": vectors[i]} for i in range(num_vectors)]

client.insert(collection_name=COLLECTION_NAME, data=entities)
# flush 保证数据落盘
client.flush(COLLECTION_NAME)

# 5. 创建索引（此时集合中已有数据）
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="vector",
    metric_type="L2",
    index_type="IVF_PQ",
    index_name="vector_index",
    params={
        "nlist": 64,  # 聚类中心数量，通常设置为 4*sqrt(n)，n 为向量数量
        "m": 32,      # 向量被分割的子向量数量，通常为 dim/m >= 2，这里 128/32=4
        "nbits": 8    # 每个子向量的编码位数，通常为 8 位
    }
)
client.create_index(
    collection_name=COLLECTION_NAME,
    index_params=index_params,
    sync=True
)

# 验证索引
print("索引列表:", client.list_indexes(collection_name=COLLECTION_NAME))
print("索引详情:", client.describe_index(
    collection_name=COLLECTION_NAME,
    index_name="vector_index"
))

# 6. load 后再搜索
client.load_collection(collection_name=COLLECTION_NAME)
latencies = []
num_search = 105  # 总共执行100次
warmup = 5        # 前5次作为warmup
for i in range(num_search):
    search_vectors = [[random.random() for _ in range(128)]]
    start_time = time.time()  # 记录开始时间
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=search_vectors,
        ann_field="vector",
        limit=5,
        output_fields=["id"],
        search_params={
            "params": {
                "nprobe": 10  # 设置搜索时检查的聚类数量
            }
        }
    )
    end_time = time.time()  # 记录结束时间
    latency_ms = (end_time - start_time) * 1000  # 转换为毫秒
    if i >= warmup:
        latencies.append(latency_ms)
if latencies:
    latencies_np = np.array(latencies)
    print("\n时延统计结果（单位：ms）：")
    print(f"MAX:{latencies_np.max():>10.2f}")
    print(f"MIN:{latencies_np.min():>10.2f}")
    print(f"AVG:{latencies_np.mean():>10.2f}")
    print(f"P50:{np.percentile(latencies_np, 50):>10.2f}")
    print(f"P99:{np.percentile(latencies_np, 99):>10.2f}")
else:
    print("无有效时延数据")

print("\n搜索结果:")
for hits in results:
    for hit in hits:
        # 注意用 dict 方式访问
        print(f"ID: {hit['id']}, 距离: {hit['distance']}")

# 清理
client.release_collection(collection_name=COLLECTION_NAME)
# client.disconnect()
