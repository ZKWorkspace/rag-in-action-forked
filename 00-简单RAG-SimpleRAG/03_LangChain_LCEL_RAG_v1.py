import os
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()
model_provider_api_key = os.getenv("O3_API_KEY")
model_provider_url_base = os.getenv("O3_URL_BASE")
print(f"Model provider api key : {model_provider_api_key}")
print(f"Model provider url base: {model_provider_url_base}")

# 1. 加载文档
# from langchain_community.document_loaders import WebBaseLoader

# loader = WebBaseLoader(
#     web_paths=("https://zh.wikipedia.org/wiki/黑神话：悟空",)
# )
# docs = loader.load()
from langchain_community.document_loaders import UnstructuredHTMLLoader
loader = UnstructuredHTMLLoader(
    "/home/zorn/Workspace/2025/培训与考试/直播_黄佳_企业落地RAG时的难点与痛点/4小时快速上手RAG/4/黑神话悟空-维基百科自由的百科全书.html"
)
docs = loader.load()

# 2. 分割文档
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 3. 设置嵌入模型
from langchain_openai import OpenAIEmbeddings

# embeddings = OpenAIEmbeddings()
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=model_provider_api_key,
    openai_api_base=model_provider_url_base,
)

# 4. 创建向量存储
from langchain_core.vectorstores import InMemoryVectorStore

vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(all_splits)

# 5. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. 创建提示模板
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
基于以下上下文，回答问题。如果上下文中没有相关信息，
请说"我无法从提供的上下文中找到相关信息"。
上下文: {context}
问题: {question}
回答:""")

# 7. 设置语言模型和输出解析器
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# llm = ChatOpenAI(model="gpt-3.5-turbo")
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

# 8. 构建 LCEL 链
# 管道式数据流像使用 Unix 命令管道 (|) 一样，将不同的处理逻辑串联在一起
chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),        
        "question": RunnablePassthrough()
    }    
    | prompt 
    | llm 
    | StrOutputParser() 
) # 查看每个阶段的输入输出

# 9. 执行查询
question = "黑神话悟空有哪些故事章节？"
response = chain.invoke(question) # 同步，可以换成异步执行
print(response)