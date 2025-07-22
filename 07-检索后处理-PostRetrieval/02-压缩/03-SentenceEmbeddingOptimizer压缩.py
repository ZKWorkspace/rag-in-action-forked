import os
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
from dotenv import load_dotenv
load_dotenv()

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding(
    model_name="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL")
)
llm_model = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL")
)
# 加载文档
documents = SimpleDirectoryReader("90-文档-Data/山西文旅").load_data()  
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
# 不使用优化的查询
print("不使用优化：")
query_engine = index.as_query_engine(llm=llm_model)
response = query_engine.query("山西省的主要旅游景点有哪些？")
print(f"答案：{response}")
# 使用优化（百分比截断）？？？Always Connection Fail？？？
print("\n使用优化（percentile_cutoff=0.5）：")
query_engine = index.as_query_engine(node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5)])
response = query_engine.query("山西省的主要旅游景点有哪些？")
print(f"答案：{response}")
# 使用优化（阈值截断）
print("\n使用优化（threshold_cutoff=0.7）：")
query_engine = index.as_query_engine(node_postprocessors=[SentenceEmbeddingOptimizer(threshold_cutoff=0.7)])
response = query_engine.query("山西省的主要旅游景点有哪些？")
print(f"答案：{response}")
