# 导入相关的库
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding# 需要pip install llama-index-embeddings-huggingface
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.deepseek import DeepSeek  # 需要pip install llama-index-llms-deepseek
from llama_index.core import Settings # 可以看看有哪些Setting

# https://docs.llamaindex.ai/en/stable/examples/llm/deepseek/
# Settings.llm = DeepSeek(model="deepseek-chat")
# Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh")
# Settings.llm = OpenAI(model="gpt-3.5-turbo")
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 加载环境变量
from dotenv import load_dotenv
import os

# 加载 .env 文件中的环境变量
load_dotenv()
model_provider_api_key = os.getenv("O3_API_KEY")
model_provider_url_base = os.getenv("O3_URL_BASE")
print(f"Model provider api key : {model_provider_api_key}")
print(f"Model provider url base: {model_provider_url_base}")

# 创建 Deepseek LLM（通过API调用最新的DeepSeek大模型）
Settings.llm = DeepSeek(
    model="deepseek-reasoner", # 使用最新的推理模型R1
    api_key=model_provider_api_key,
    api_base=model_provider_url_base,
)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=model_provider_api_key,
    api_base=model_provider_url_base,
)

# 加载数据
print(f"Loading data from: 90-文档-Data/黑悟空/设定.txt")
try:
    documents = SimpleDirectoryReader(input_files=["90-文档-Data/黑悟空/设定.txt"]).load_data() 
except Exception as e:
    print(f"Error loading data: {e}")
    documents = []

# 构建索引
print(f"Building index...")
index = VectorStoreIndex.from_documents(documents)
all_nodes = index.docstore.docs
print(f"{len(all_nodes)} nodes have been built:")
for i, (node_id, node) in enumerate(all_nodes.items(), 1):
    print(f"Node ID: {node_id}")
    print(f"Content: {node.text}")
    print("-" * 100)

# 创建问答引擎
print(f"Creating query engine...")
query_engine = index.as_query_engine()

# 开始问答
print(f"Starting query...")
# question = "黑神话悟空中有哪些故事章节?如果你不清楚这个问题，请用简短且幽默的中文回答我"
question = "黑神话悟空中有哪些战斗工具?"
print(f"\nQ: {question}")
response = query_engine.query(question)
print(f"\nA: {response}")