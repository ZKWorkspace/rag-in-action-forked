import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # pip install faiss-cpu
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
model_provider_api_key = os.getenv("O3_API_KEY")
model_provider_url_base = os.getenv("O3_BASE_URL")

# 初始化 OpenAI 模型
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=model_provider_api_key,
    base_url=model_provider_url_base)
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=model_provider_api_key,
    base_url=model_provider_url_base,
    temperature=0)

# 加载 PDF 文件
loader = UnstructuredPDFLoader("90-文档-Data/复杂PDF/uber_10q_march_2022.pdf")
documents = loader.load()

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)
chunks = text_splitter.split_documents(documents)

# 创建向量存储
vectorstore = FAISS.from_documents(chunks, embeddings)

# 创建检索链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    verbose=True
)

# 查询示例
query = "What is the change of free cash flow and what is the rate from the financial and operational highlights?"
# query = "how many COVID-19 response initiatives in year 2021?"
response = qa_chain.invoke({"query": query})

print("\n************LangChain Query Response************")
print("Answer:", response["result"])
print("\nSource Documents:")
for i, doc in enumerate(response["source_documents"], 1):
    print(f"\nDocument {i}:")
    print(doc.page_content[:200] + "...")