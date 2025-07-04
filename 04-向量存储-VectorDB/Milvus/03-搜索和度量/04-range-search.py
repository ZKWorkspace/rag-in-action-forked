from pymilvus import MilvusClient, DataType
import random
import numpy as np

# 1. 设置 Milvus 客户端
import os
from dotenv import load_dotenv

load_dotenv()
client = MilvusClient(uri="http://172.17.19.130:19530", token=os.getenv("MILVUS_TOKEN"))
COLLECTION_NAME = "coll_04_ann_search_range"

# 如果集合已存在，则删除
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

# 2. 创建 schema
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)
schema.add_field(field_name="color", datatype=DataType.VARCHAR, max_length=100)

# 3. 创建集合
client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

# 4. 插入随机向量数据
def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

num_vectors = 1024
vectors = [[random.random() for _ in range(128)] for _ in range(num_vectors)]
ids = list(range(num_vectors))
colors = [f"color_{random.randint(1, 1000)}" for _ in range(num_vectors)]
entities = [{"id": ids[i], "vector": normalize_vector(vectors[i]), "color": colors[i]} for i in range(num_vectors)]

client.insert(collection_name=COLLECTION_NAME, data=entities)

# 5. 创建索引
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="vector",
    metric_type="L2",
    index_type="FLAT",
    index_name="vector_index",
    params={}
)
client.create_index(
    collection_name=COLLECTION_NAME,
    index_params=index_params,
    sync=True
)

# 6. 加载集合
client.flush(collection_name=COLLECTION_NAME)
client.load_collection(collection_name=COLLECTION_NAME)

# 7. 单向量搜索示例
print("\n=== 单向量搜索 ===")
query_vector = [random.random() for _ in range(128)]
normalized_query_vector = normalize_vector(query_vector)
results = client.search(
    collection_name=COLLECTION_NAME,
    data=[normalized_query_vector],
    anns_field="vector",
    limit=3,
    search_params={"metric_type": "L2"}
)

print("搜索结果:")
for hits in results:
    for hit in hits:
        print(f"ID: {hit['id']}, 距离: {hit['distance']}")

# 8. 批量向量搜索示例
print("\n=== 批量向量搜索 ===")
query_vectors = [[random.random() for _ in range(128)] for _ in range(2)]
normalized_query_vectors = [normalize_vector(vec) for vec in query_vectors]
results = client.search(
    collection_name=COLLECTION_NAME,
    data=normalized_query_vectors,
    anns_field="vector",
    limit=3,
    search_params={"metric_type": "L2"}
)

print("批量搜索结果:")
for i, hits in enumerate(results):
    print(f"\n查询向量 {i+1} 的结果:")
    for hit in hits:
        print(f"ID: {hit['id']}, 距离: {hit['distance']}")

# 9. 带输出字段的搜索示例
print("\n=== 带输出字段的搜索 ===")
results = client.search(
    collection_name=COLLECTION_NAME,
    data=[normalized_query_vector],
    anns_field="vector",
    limit=3,
    search_params={"metric_type": "L2"},
    output_fields=["color"]
)

print("带输出字段的搜索结果:")
for hits in results:
    for hit in hits:
        print(f"ID: {hit['id']}, 距离: {hit['distance']}, 颜色: {hit['entity']['color']}")

# 10. 范围搜索示例
print("\n=== 范围搜索 ===")
# 使用 L2 距离度量，设置范围搜索参数
# 注意：对于 L2 距离，range_filter 应该小于 radius
results = client.search(
    collection_name=COLLECTION_NAME,
    data=[normalized_query_vector],
    anns_field="vector",
    limit=10,  # 增加限制以显示更多结果
    search_params={
        "metric_type": "L2",
        "params": {
            "radius": 1.0,  # 外圈半径，定义了一个最大距离阈值，超过这个阈值的向量不会被返回
            "range_filter": 0.5  # 内圈半径，定义了一个最小距离阈值，小于这个阈值的向量不会被返回
        }
    },
    output_fields=["color"]
)

print("范围搜索结果:")
print(f"搜索范围: 距离在 {0.5} 到 {1.0} 之间的向量")
for hits in results:
    for hit in hits:
        print(f"ID: {hit['id']}, 距离: {hit['distance']}, 颜色: {hit['entity']['color']}")

# 11. 清理
client.release_collection(collection_name=COLLECTION_NAME)
