# 1. 加载文档
import os
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()
model_provider_api_key = os.getenv("O3_API_KEY")
model_provider_url_base = os.getenv("O3_URL_BASE")
print(f"Model provider api key : {model_provider_api_key}")
print(f"Model provider url base: {model_provider_url_base}")

# from langchain_community.document_loaders import WebBaseLoader
# loader = WebBaseLoader(
#     # web_paths=("https://zh.wikipedia.org/wiki/黑神话：悟空",)
#     web_paths=("https://baike.baidu.com/item/黑神话：悟空",)
# )

from langchain_community.document_loaders import UnstructuredHTMLLoader
loader = UnstructuredHTMLLoader(
    "/home/zorn/Workspace/2025/培训与考试/直播_黄佳_企业落地RAG时的难点与痛点/4小时快速上手RAG/4/黑神话悟空-维基百科自由的百科全书.html"
)
docs = loader.load()
print(f"Web page content:\n{docs}")
print("-" * 100)

# 2. 文档分块
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
print(f"{len(all_splits)} content slices after splitting")
for i, split in enumerate(all_splits, 1):
    print(f"Slice ID: {i}")
    print(f"Content : {split.page_content}")
    print("-" * 100)

# 3. 设置嵌入模型
# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-zh-v1.5",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=model_provider_api_key,
    openai_api_base=model_provider_url_base,
)

# 4. 创建向量存储
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# 5. 构建用户查询
question = "黑神话悟空中有哪些故事章节?"
print(f"Q: {question}")

# 6. 在向量存储中搜索相关文档，并准备上下文内容
retrieved_docs = vector_store.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
print(f"R: {docs_content}")

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
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-reasoner",  # DeepSeek API 支持的模型名称
    # base_url="https://api.deepseek.com/v1",
    base_url=model_provider_url_base,
    temperature=0.7,        # 控制输出的随机性(0-1之间,越大越随机)
    max_tokens=2048,        # 最大输出长度
    top_p=0.95,            # 控制输出的多样性(0-1之间)
    presence_penalty=0.0,   # 重复惩罚系数(-2.0到2.0之间)
    frequency_penalty=0.0,  # 频率惩罚系数(-2.0到2.0之间)
    # api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量加载API key
    api_key=model_provider_api_key,
)
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(f"A: {answer.content}")


