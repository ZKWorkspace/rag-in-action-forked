# 导入相关的库
import os
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
from dotenv import load_dotenv
load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain.load import dumps, loads

"""
RRF（Reciprocal Rank Fusion）重排算法实现

RRF是一种简单而有效的多检索结果融合算法，它通过将多个检索查询的结果进行排名融合，
来提高检索的准确性和覆盖面。

核心思想：
1. 对于同一个用户问题，生成多个不同角度的查询
2. 分别对每个查询进行检索
3. 使用RRF算法将多个检索结果列表融合成一个统一的排序列表
4. RRF算法为每个文档分配分数：score = 1/(rank + k)，其中rank是该文档在某个结果列表中的排名

优势：
- 提高检索的覆盖面：多个查询可以从不同角度检索相关文档
- 降低单一查询的偏差：通过多查询融合减少单一查询的局限性
- 简单高效：算法复杂度低，易于实现和理解
"""

# 文档目录配置
doc_dir = "90-文档-Data/山西文旅"

def load_documents(directory):
    """
    文档加载函数
    
    功能：读取指定目录中的所有文档（支持PDF、TXT格式）
    
    参数：
        directory (str): 文档所在目录路径
    
    返回：
        list: 加载的文档列表，每个文档包含内容和元数据
    
    说明：
        - 遍历目录中的所有文件
        - 根据文件扩展名选择合适的加载器
        - 支持PDF和TXT格式文件
        - 跳过不支持的文件格式
    """
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if filename.endswith(".pdf"):
            # 使用PyPDFLoader加载PDF文件
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            # 使用TextLoader加载TXT文件
            loader = TextLoader(filepath)
        else:
            continue  # 跳过不支持的文件类型
        
        # 加载文档并添加到列表中
        documents.extend(loader.load())
    return documents

# 第一步：加载文档
print("📖 正在加载文档...")
docs = load_documents(doc_dir)
print(f"✅ 成功加载 {len(docs)} 个文档")

# 第二步：文本切块（分割）
print("\n🔪 正在进行文本切块...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # 每个文本块的最大字符数
    chunk_overlap=50     # 相邻文本块之间的重叠字符数，确保上下文连续性
)
splits = text_splitter.split_documents(docs)
print(f"✅ 文档已切分为 {len(splits)} 个文本块")

# 第三步：创建向量索引
print("\n🔍 正在创建向量索引...")
# 使用HuggingFace的轻量级嵌入模型
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# 使用Chroma向量数据库存储文档向量
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
# 创建检索器
retriever = vectorstore.as_retriever()
print("✅ 向量索引创建完成")

def reciprocal_rank_fusion(results: list[list], k=60):
    """
    RRF（Reciprocal Rank Fusion）算法实现
    
    功能：将多个检索结果列表融合成一个统一的排序列表
    
    参数：
        results (list[list]): 多个检索结果列表，每个列表包含按相关性排序的文档
        k (int): RRF算法的调节参数，默认值60（经验值）
    
    返回：
        list: 融合后的(文档, 分数)元组列表，按分数降序排序
    
    算法原理：
        1. 对于每个检索结果列表中的每个文档
        2. 计算该文档的RRF分数：score = 1 / (rank + k)
        3. 如果同一文档出现在多个列表中，累加其分数
        4. 按最终分数对所有文档进行排序
    
    优势：
        - rank越小（排名越靠前），分数越高
        - k参数防止分母为0，并调节不同排名之间的差距
        - 多次出现的文档会获得更高的累积分数
    """
    print(f"🔄 RRF算法处理 {len(results)} 个检索结果列表...")
    
    fused_scores = {}  # 存储每个文档的累积分数
    
    # 遍历每个检索结果列表
    for list_idx, docs in enumerate(results):
        print(f"  处理第 {list_idx + 1} 个结果列表，包含 {len(docs)} 个文档")
        
        # 遍历该列表中的每个文档
        for rank, doc in enumerate(docs):
            # 将文档序列化为字符串作为唯一标识
            doc_str = dumps(doc)
            
            # 如果该文档首次出现，初始化分数
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            
            # 计算RRF分数并累加
            rrf_score = 1 / (rank + k)
            fused_scores[doc_str] += rrf_score
            
            # 调试信息：显示文档在当前列表中的排名和分数
            if rank < 3:  # 只显示前3个文档的详细信息
                print(f"    文档 {rank+1}: RRF分数 = 1/({rank}+{k}) = {rrf_score:.4f}")
    
    # 按分数降序排序，返回(文档, 分数)元组列表
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    print(f"✅ RRF融合完成，共 {len(reranked_results)} 个唯一文档")
    return reranked_results

# 第四步：多查询生成
print("\n💭 配置多查询生成器...")
template = """你是一个帮助用户生成多个搜索查询的助手。

请根据以下问题生成4个不同角度的相关搜索查询，这些查询应该：
1. 从不同的角度理解原问题
2. 使用不同的关键词和表达方式
3. 覆盖问题的不同方面
4. 输出的每一行表示一个角度的搜索查询，不同角度按行分隔

原问题：{question}

请生成4个相关的搜索查询："""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)
llm = ChatDeepSeek(
    model="deepseek-ai/DeepSeek-V3",
    api_base=os.getenv("SILICON_FLOW_BASE_URL"),
    api_key=os.getenv("SILICON_FLOW_API_KEY")
)

# 创建查询生成链
generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))  # 按行分割生成的查询
)
print("✅ 多查询生成器配置完成")

# 第五步：测试示例
print("\n🎯 开始RRF重排测试...")
questions = [
    "山西有哪些著名的旅游景点？",
    # "云冈石窟的历史背景是什么？",
    # "五台山的文化和宗教意义是什么？"
]

# 对每个问题进行RRF检索和重排
for idx, question in enumerate(questions, 1):
    print(f"\n{'='*50}")
    print(f"🔍 第 {idx} 个问题：{question}")
    print('='*50)
    
    # 第一步：生成多个查询
    print("\n1️⃣ 生成多个相关查询...")
    queries = generate_queries.invoke({"question": question})
    # 过滤空查询
    queries = [q.strip() for q in queries if q.strip()]
    print(f"生成了 {len(queries)} 个查询：")
    for i, query in enumerate(queries, 1):
        print(f"  查询 {i}: {query}")
    
    # 第二步：对每个查询进行检索
    print(f"\n2️⃣ 对每个查询进行向量检索...")
    all_results = []
    for i, query in enumerate(queries, 1):
        print(f"  检索查询 {i}: {query}")
        docs = retriever.invoke(query)
        all_results.append(docs)
        print(f"    检索到 {len(docs)} 个相关文档")
    
    # 第三步：使用RRF算法融合结果
    print(f"\n3️⃣ 使用RRF算法融合检索结果...")
    reranked_docs = reciprocal_rank_fusion(all_results)
    
    # 第四步：展示最终结果
    print(f"\n4️⃣ 最终RRF重排结果（显示前3个）：")
    print(f"总共融合了 {len(reranked_docs)} 个唯一文档")
    
    for i, (doc, score) in enumerate(reranked_docs[:3], 1):
        print(f"\n📄 排名 {i} (RRF分数: {score:.4f}):")
        # 截取前200个字符避免输出过长
        content_preview = doc.page_content[:200].replace('\n', ' ').strip()
        print(f"   内容预览: {content_preview}...")
        
        # 显示文档来源信息（如果有）
        if hasattr(doc, 'metadata') and doc.metadata:
            source = doc.metadata.get('source', '未知来源')
            print(f"   来源: {source}")

print(f"\n🎉 RRF重排测试完成！")
print("\n📋 RRF算法总结：")
print("- ✅ 多角度查询生成：从不同角度理解用户问题")
print("- ✅ 多检索结果融合：整合多个检索结果的优势")
print("- ✅ 排名优化：通过RRF算法重新排序文档")
print("- ✅ 提高召回率：减少单一查询的遗漏")
print("- ✅ 提升相关性：多次出现的文档获得更高权重")
