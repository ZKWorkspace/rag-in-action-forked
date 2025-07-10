# ingest_q2sql.py
import os
from dotenv import load_dotenv
load_dotenv()
import logging
from pymilvus import MilvusClient, MilvusException, DataType, FieldSchema, CollectionSchema
from pymilvus import model
import torch
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 初始化嵌入函数
embedding_function = model.dense.OpenAIEmbeddingFunction(
    model_name='BAAI/bge-m3',
    base_url=os.getenv("SILICON_FLOW_BASE_URL"),
    api_key=os.getenv("SILICON_FLOW_API_KEY")
)

# 2. 加载 Q->SQL 对（假设 q2sql_pairs.json 数组，每项 { "question": ..., "sql": ... }）
with open("90-文档-Data/sakila/q2sql_pairs.json", "r") as f:
    pairs = json.load(f)
    logging.info(f"[Q2SQL] 从JSON文件加载了 {len(pairs)} 个问答对")

# 3. 连接 Milvus
client = MilvusClient(uri="http://172.17.19.130:19530", token=os.getenv("MILVUS_TOKEN"))
DATABASE_NAME = "sakila"
if DATABASE_NAME not in client.list_databases():
    try:
        client.create_database(db_name=DATABASE_NAME)
        logging.info(f"✓ {DATABASE_NAME} 创建成功")
    except MilvusException as e:
        logging.error(str(e))
        exit
client.use_database(db_name=DATABASE_NAME)

# 4. 定义 Collection Schema
vector_dim = len(embedding_function(["dummy"])[0])
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="sql_text", dtype=DataType.VARCHAR, max_length=2000),
]
schema = CollectionSchema(fields, description="Q2SQL Knowledge Base", enable_dynamic_field=False)

# 5. 创建 Collection（如不存在）
collection_name = "q2sql_knowledge"
if client.has_collection(collection_name):
    client.drop_collection(collection_name=collection_name)
    logging.info(f"[DDL] 删除表 {collection_name} 成功")
client.create_collection(collection_name=collection_name, schema=schema)
logging.info(f"✓ 表{collection_name} 创建成功")

# 6. 为向量字段添加索引
index_params = client.prepare_index_params()
index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE", params={})
client.create_index(collection_name=collection_name, index_params=index_params)

# 7. 批量插入 Q2SQL 对
data = []
texts = []
for pair in pairs:
    texts.append(pair["question"])
    data.append({"question": pair["question"], "sql_text": pair["sql"]})

logging.info(f"[Q2SQL] 准备处理 {len(data)} 个问答对")

# 生成全部嵌入
embeddings = embedding_function(texts)
logging.info(f"[Q2SQL] 成功生成了 {len(embeddings)} 个向量嵌入")

# 组织为 Milvus insert 格式
records = []
for emb, rec in zip(embeddings, data):
    rec["vector"] = emb
    records.append(rec)

res = client.insert(collection_name=collection_name, data=records)
client.flush(collection_name=collection_name)
logging.info(f"[Q2SQL] 成功插入了 {len(records)} 条记录到Milvus")
logging.info(f"[Q2SQL] 插入结果: {res}")

logging.info("[Q2SQL] 知识库构建完成")
