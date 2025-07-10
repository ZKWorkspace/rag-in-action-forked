import os
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
import logging
from langchain.retrievers import RePhraseQueryRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 设置日志记录
logging.basicConfig()
logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.INFO)
# 加载游戏文档数据
loader = TextLoader("90-文档-Data/黑悟空/设定.txt", encoding='utf-8')
data = loader.load()
# 文本分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
# 创建向量存储
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(documents=all_splits, embedding= embed_model)
# 设置RePhraseQueryRetriever
llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=0)
retriever_from_llm = RePhraseQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm # 使用DeepSeek模型做重写器
)
# 示例输入：游戏相关查询
query = "那个，我刚开始玩这个游戏，感觉很难，在普陀山那一关，嗯，怎么也过不去。先学什么技能比较好？新手求指导！"
# 调用RePhraseQueryRetriever进行查询重写
# docs是检索结果，在retriever_from_llm的执行过程中，可以看到INFO日志打印:
docs = retriever_from_llm.invoke(query)
print(docs)
# INFO:langchain.retrievers.re_phraser:Re-phrased question: Here’s the refined query for vectorstore retrieval, 
# focusing on the core intent and stripping irrelevant conversational elements:  
#
# **"普陀山关卡新手推荐技能"**  
#
# Key elements preserved:  
# - Game context (普陀山关卡)  
# - User status (新手)  
# - Core need (推荐技能)  
#
# Removed:  
# - Subjective difficulty ("感觉很难")  
# - Conversational filler ("那个"，"嗯")  
# - Indirect requests ("先学什么比较好"，"求指导" condensed into "推荐") 
#
# "普陀山关卡新手推荐技能"就是经过重写后的查询语句