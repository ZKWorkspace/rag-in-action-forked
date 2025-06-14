# 1. 加载文档
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader # pip install langchain-community

loader = WebBaseLoader(
    web_paths=("https://zh.wikipedia.org/wiki/黑神话：悟空",)
)
docs = loader.load()

# 2. 文档分块
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 打印所有分割后的文本内容
print("\n=== 分割后的文本内容 ===")
print(f"总共分割成 {len(all_splits)} 个文本块")
for i, split in enumerate(all_splits, 1):
    print(f"\n--- 第 {i} 个文本块 ---")
    print(f"内容:\n{split.page_content}")
    print("-" * 100)

# 3. 设置嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings # pip install langchain-huggingface

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 4. 创建向量存储
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# 5. 构建用户查询
question = "请列举黑神话所取得的所有奖项？"

# 6. 在向量存储中搜索相关文档，并准备上下文内容
retrieved_docs = vector_store.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 7. 构建提示模板
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
                基于以下上下文，回答问题。如果上下文中没有相关信息，
                请说"我无法从提供的上下文中找到相关信息"。
                上下文: {context}
                问题: {question}
                回答:"""
                                          )

# 8. 使用大语言模型生成答案
from langchain_deepseek import ChatDeepSeek # pip install langchain-deepseek

llm = ChatDeepSeek(
    model="deepseek-chat",  # DeepSeek API 支持的模型名称
    temperature=0.7,        # 控制输出的随机性
    max_tokens=2048,        # 最大输出长度
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量加载API key
)
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)


