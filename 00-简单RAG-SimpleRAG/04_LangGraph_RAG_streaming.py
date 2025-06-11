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

# 2. 文档分块
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 3. 设置嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 4. 创建向量存储aa
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# 5. 定义RAG提示词
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

# 6. 定义应用状态
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# 7. 定义检索步骤
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

# 8. 定义生成步骤
def generate(state: State):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="gpt-4o",
        base_url=model_provider_url_base,
        api_key=model_provider_api_key,
    )
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# 9. 构建和编译应用
from langgraph.graph import START, StateGraph # pip install langgraph
graph = (
    StateGraph(State)
    .add_sequence([retrieve, generate])
    .add_edge(START, "retrieve")
    .compile()
)

# 10. 运行查询
question = "黑神话悟空有哪些故事章节？"
print(f"\n问题: {question}")
# response = graph.invoke({"question": question})
# print(f"\n答案: {response['answer']}")
print("\n答案:")
for message, metadata in graph.stream(
    {"question": question}, stream_mode="messages"
):
    print(message.content, end="|")