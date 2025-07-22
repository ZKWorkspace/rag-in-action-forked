import os
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank # pip install rank_llm
import torch

"""
RankLLM重排算法实现

RankLLM是一种基于大语言模型（LLM）的重排方法，利用LLM强大的语言理解能力进行文档重排。

核心原理：
1. 利用LLM的深度语言理解能力判断查询与文档的相关性
2. 通过prompt engineering引导LLM进行排序决策
3. 结合LLM的推理能力，能够处理复杂的语义关系

技术特点：
- 语义理解深度：基于LLM的强大语言理解能力
- 推理能力强：能够进行复杂的逻辑推理和语义匹配
- 灵活性高：可以通过prompt调整适应不同领域和任务
- 解释性好：LLM可以提供排序的理由和解释

与其他方法对比：
- vs BERT类模型：语义理解更深入，能够处理更复杂的推理
- vs 传统重排：能够理解上下文和隐含信息
- vs 嵌入模型：不仅考虑相似度，还考虑逻辑关系

适用场景：
- 对精度要求极高的应用
- 需要复杂推理的查询
- 领域专业性强的文档检索
- 需要可解释性的重排任务

注意事项：
- 计算成本较高（调用LLM API）
- 延迟相对较大
- 需要合理设计prompt
"""

print("🔄 初始化RankLLM重排系统...")

# 1. 文档加载和预处理
print("📖 加载和预处理文档...")
doc_path = "90-文档-Data/山西文旅/云冈石窟.txt"
print(f"文档路径: {doc_path}")

print("  🔤 使用TextLoader加载文档...")
documents = TextLoader(doc_path).load()
print(f"  ✅ 成功加载文档，原始文档数量: {len(documents)}")

print("  ✂️  开始文档分割...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 每个文档块500个字符
    chunk_overlap=100     # 块之间重叠100个字符，保持上下文连续性
)
texts = text_splitter.split_documents(documents)
print(f"  📊 分割后文档块数量: {len(texts)}")

# 为每个文档块添加唯一ID
print("  🆔 为文档块添加唯一标识...")
for idx, text in enumerate(texts):
    text.metadata["id"] = idx
    text.metadata["chunk_size"] = len(text.page_content)
print("  ✅ 文档预处理完成")

# 2. 创建向量检索器
print(f"\n🔍 创建FAISS向量检索器...")
print("  📥 加载中文嵌入模型...")
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")  # 使用中文优化的嵌入模型
print("  🏗️  构建FAISS向量索引...")
retriever = FAISS.from_documents(texts, embed_model).as_retriever(
    search_kwargs={"k": 20}  # 第一阶段检索Top-20文档
)
print(f"  ✅ 向量检索器创建完成，将返回Top-20候选文档")

# 3. GPU内存优化（如果使用GPU）
print(f"\n🧹 优化GPU内存使用...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("  🗑️  已清理GPU缓存")
else:
    print("  💻 当前使用CPU模式")

# 4. 配置RankLLM重排器
print(f"\n🤖 配置RankLLM重排器...")
print("  ⚙️  RankLLM配置参数:")
print("    - top_n: 3 (最终返回前3个文档)")
print("    - model: gpt (使用GPT模型)")
print("    - gpt_model: gpt-4o-mini (高效的GPT模型)")

# 配置OPENAI 代理信息
# OPENAI_BASE_URL = "https://vip.apiyi.com/v1"
# OPENAI_API_KEY = ""
# os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

compressor = RankLLMRerank(
    top_n=3,                    # 最终返回前3个最相关的文档
    model="gpt",                # 使用GPT模型进行重排
    gpt_model="gpt-4o-mini"     # 选择高效的GPT-4o-mini模型
)
print("  ✅ RankLLM重排器配置完成")

# 5. 创建上下文压缩检索器
print(f"\n🔗 创建上下文压缩检索器...")
print("  📋 组合向量检索器和RankLLM重排器...")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,     # 使用RankLLM作为压缩器（重排器）
    base_retriever=retriever        # 使用FAISS作为基础检索器
)
print("  ✅ 检索链条构建完成：FAISS检索 → RankLLM重排")

# 6. 执行查询和重排
print(f"\n🎯 开始执行查询和重排...")
query = "云冈石窟有哪些著名的造像？"
print(f"查询问题: {query}")

print(f"\n第一阶段 - FAISS向量检索:")
print("  🔍 基于语义相似度检索候选文档...")

print(f"\n第二阶段 - RankLLM重排:")
print("  🤖 调用GPT模型进行深度语义重排...")
print("  ⏳ 正在处理中（LLM推理需要一些时间）...")

try:
    compressed_docs = compression_retriever.invoke(query)
    print(f"  ✅ RankLLM重排完成")
    print(f"  📊 最终返回 {len(compressed_docs)} 个高质量文档")

    # 7. 格式化输出重排结果
    def pretty_print_docs(docs):
        """
        美化文档输出函数
        
        功能：以易读的格式展示重排后的文档结果
        
        参数：
            docs (list): 重排后的文档列表
        """
        print(f"\n{'='*60}")
        print(f"🏆 RankLLM重排最终结果")
        print(f"{'='*60}")
        print(f"查询: {query}")
        print(f"重排后文档（按相关性降序）:")
        
        result_parts = []
        for i, doc in enumerate(docs, 1):
            doc_info = f"\n📄 排名 {i}:\n"
            doc_info += f"   文档内容:\n{doc.page_content}\n"
            
            # 显示文档元数据
            if hasattr(doc, 'metadata') and doc.metadata:
                doc_info += f"   文档ID: {doc.metadata.get('id', '未知')}\n"
                doc_info += f"   内容长度: {doc.metadata.get('chunk_size', len(doc.page_content))} 字符\n"
                if 'source' in doc.metadata:
                    doc_info += f"   来源文件: {doc.metadata['source']}\n"
            
            result_parts.append(doc_info)
        
        return "\n" + ("-" * 100) + "\n".join(result_parts)

    # 输出美化的结果
    formatted_result = pretty_print_docs(compressed_docs)
    print(formatted_result)

except Exception as e:
    print(f"  ❌ RankLLM重排失败: {str(e)}")
    print("  💡 可能的原因:")
    print("    - GPT API密钥未配置或无效")
    print("    - 网络连接问题")
    print("    - API配额已用完")
    print("    - 文档内容格式问题")
    print("  🔧 建议检查:")
    print("    - OpenAI API密钥配置")
    print("    - 网络连接状态")
    print("    - 文档文件是否存在")

# 8. 资源清理
print(f"\n🧹 清理系统资源...")
try:
    # 清理RankLLM模型（如果需要）
    if 'compressor' in locals():
        del compressor
        print("  🗑️  已释放RankLLM模型资源")
    
    # 再次清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("  🗑️  已清理GPU缓存")
    
    print("  ✅ 资源清理完成")
except Exception as e:
    print(f"  ⚠️  资源清理时出现警告: {str(e)}")

print(f"\n📋 RankLLM重排总结:")
print("- ✅ 深度理解：基于LLM的强大语言理解能力")
print("- ✅ 推理能力：能够进行复杂的逻辑推理和语义匹配")
print("- ✅ 高精度：利用最先进的语言模型技术")
print("- ✅ 可解释：LLM可以提供排序的理由和依据")
print("- ⚠️  高成本：需要调用LLM API，成本相对较高")
print("- ⚠️  高延迟：LLM推理时间相对较长")
print("- 💡 最佳实践：适用于对精度要求极高的重要查询")
print("- 🔧 优化建议：合理设计prompt以提升重排效果")
