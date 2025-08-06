#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
上下文检索（Contextual Retrieval）与Milvus实现
基于Anthropic提出的方法，解决传统RAG中语义隔离问题

核心概念：
1. 传统RAG问题：文档被分割成独立的块，丢失了上下文信息
2. 上下文检索解决方案：为每个块添加文档上下文，使其语义更完整
3. 深度评估（Deep Evaluation）：多维度评估检索系统的性能

技术栈：
- Milvus: 向量数据库，支持密集向量和稀疏向量
- SentenceTransformer: 生成文本的向量表示
- OpenAI GPT: 用于生成上下文化的文本块（替代Claude）
- Cohere Reranker: 重排序模型，优化检索结果

API变更说明：
- 原版本使用Anthropic Claude API进行上下文增强
- 当前版本改用OpenAI GPT API，提供更好的可用性和性能
- Claude相关代码已注释，可根据需要切换回来

===============================================================================
📋 代码结构分析 (Code Structure Analysis)
===============================================================================

🏗️ 整体架构设计:
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG上下文检索系统架构                          │
├─────────────────────────────────────────────────────────────────────┤
│  输入层: 原始文档 → 文档分块 → 上下文增强 → 向量化 → 存储           │
│  检索层: 查询向量化 → 相似度搜索 → 重排序 → 结果返回              │
│  评估层: 黄金标准对比 → 性能指标计算 → 结果分析                    │
└─────────────────────────────────────────────────────────────────────┘

🔧 模块职责划分:
1. MilvusContextualRetriever (核心检索器)
   - 负责向量数据库的操作
   - 实现多种检索策略
   - 管理上下文化和重排序流程

2. 评估模块 (Performance Evaluation)
   - evaluate_retrieval(): 核心评估逻辑
   - evaluate_db(): 数据库性能评估
   - retrieve_base(): 基础检索接口

3. 数据处理模块 (Data Processing)
   - download_data(): 数据下载
   - load_jsonl(): 数据加载
   - 数据格式标准化

4. 实验控制模块 (Experiment Control)
   - main(): 实验流程控制
   - 三种检索策略对比
   - 性能指标统计

===============================================================================
🔄 数据流程分析 (Data Flow Analysis)
===============================================================================

📊 数据处理流程:
原始文档 → 文档分块 → [可选]上下文增强 → 向量化 → Milvus存储
    ↓
查询输入 → 查询向量化 → 相似度搜索 → [重排序] → 结果输出
    ↓
评估对比 → 性能指标 → 结果分析

🎯 检索策略对比:
┌──────────────┬──────────────┬──────────────┬──────────────┐
│   策略类型    │   数据预处理  │   检索方法    │   后处理     │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 标准检索     │ 原始文本块   │ 密集向量搜索 │ 无           │
│ 上下文检索   │ LLM增强块    │ 密集向量搜索 │ 无           │
│ 重排序检索   │ LLM增强块    │ 密集向量搜索 │ Cohere重排序 │
└──────────────┴──────────────┴──────────────┴──────────────┘

🔍 评估体系:
输入: 查询 + 黄金标准答案
处理: 检索 → 匹配 → 计分
输出: Pass@K分数、平均分数、召回率

===============================================================================
⚡ 执行流程分析 (Execution Flow Analysis)
===============================================================================

🚀 主要执行步骤:
1️⃣ 环境初始化
   - 加载API密钥和配置
   - 初始化嵌入模型和重排序模型
   - 下载示例数据

2️⃣ 数据准备
   - 加载文档数据集
   - 创建评估查询集
   - 数据格式验证

3️⃣ 实验一: 标准检索基线
   - 创建标准Milvus集合
   - 插入原始文本块
   - 执行检索评估

4️⃣ 实验二: 上下文检索
   - 创建上下文Milvus集合
   - LLM增强文本块
   - 插入增强后的文本块
   - 执行检索评估

5️⃣ 实验三: 重排序检索
   - 使用上下文检索器
   - 启用Cohere重排序
   - 执行检索评估

6️⃣ 结果分析
   - 对比三种策略性能
   - 计算性能提升幅度
   - 生成分析报告

===============================================================================
🎛️ 关键参数配置 (Key Parameters)
===============================================================================

📋 向量数据库配置:
- 集合名称: 区分不同实验的数据集合
- 向量维度: 由嵌入模型决定（如BGE-large-zh为1024维）
- 索引类型: FLAT（精确搜索）+ SPARSE_INVERTED_INDEX（稀疏向量）
- 距离度量: 内积（IP）

🤖 LLM配置:
- 模型: gpt-3.5-turbo（快速经济型）或gpt-4（高质量）
- 最大tokens: 1000
- 温度: 0（确保一致性）
- API: OpenAI ChatGPT API

🔄 检索配置:
- 检索数量K: 默认5（Pass@5评估）
- 搜索参数: nprobe=10
- 重排序: Cohere Rerank API

===============================================================================
"""

# 导入必要的库
import os
import dotenv
dotenv.load_dotenv()
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import CohereRerankFunction
from pymilvus.model.reranker import BGERerankFunction

from typing import List, Dict, Any
from typing import Callable
from pymilvus import (
    MilvusClient,
    DataType,
    AnnSearchRequest,
    RRFRanker,
)
from tqdm import tqdm
import json
# import anthropic  # 注释掉Claude API，改用OpenAI
import openai  # 新增OpenAI API支持

class MilvusContextualRetriever:
    """
    Milvus上下文检索器类
    
    🏛️ 架构设计:
    这个类是整个检索系统的核心，采用模块化设计，支持多种检索策略的灵活组合。
    
    📦 功能模块:
    1. 标准检索：基于原始文本块的向量检索
    2. 混合检索：结合密集向量和稀疏向量的检索
    3. 上下文检索：使用LLM丰富文本块的上下文信息后进行检索
    4. 重排序检索：在检索结果基础上使用专门的重排序模型优化结果
    
    🔄 数据流向:
    文本输入 → [上下文增强] → 向量化 → Milvus存储
           ↓
    查询输入 → 向量化 → 相似度搜索 → [重排序] → 结果输出
    
    🎯 设计原则:
    - 单一职责：每个方法负责一个特定功能
    - 开放封闭：支持扩展新的检索策略
    - 依赖注入：通过构造函数注入依赖组件
    - 配置驱动：通过参数控制不同功能的启用
    
    支持标准检索、混合检索、上下文检索和重排序功能
    """
    
    def __init__(
        self,
        uri="milvus.db",
        token="",
        collection_name="contexual_bgem3",
        dense_embedding_function=None,
        use_sparse=False,
        sparse_embedding_function=None,
        use_contextualize_embedding=False,
        llm_client=None,  # 改用通用LLM客户端名称（支持OpenAI）
        use_reranker=False,
        rerank_function=None,
    ):
        """
        初始化检索器
        
        🔧 初始化流程:
        1. 设置Milvus连接参数
        2. 配置嵌入函数（密集+稀疏）
        3. 设置LLM客户端（用于上下文增强）
        4. 配置重排序功能
        5. 参数验证和错误处理
        
        参数说明：
            uri: Milvus服务地址
                - 本地文件模式：如 "./milvus.db"（Milvus Lite）
                - 服务器模式：如 "http://localhost:19530"（独立Milvus服务）
                - 云服务模式：Zilliz Cloud的连接地址
            collection_name: 集合名称，类似于数据库中的表名
            dense_embedding_function: 密集向量嵌入函数
                - 用于将文本转换为高维稠密向量（如768维、1024维等）
                - 通常使用预训练的语言模型如BGE、SentenceTransformer等
            use_sparse: 是否使用稀疏向量
                - 稀疏向量类似于TF-IDF，主要捕获关键词匹配信息
                - 与密集向量结合可以提高检索的准确性
            sparse_embedding_function: 稀疏向量嵌入函数
            use_contextualize_embedding: 是否使用上下文嵌入
                - 这是核心功能：使用LLM为每个文本块添加文档上下文
                - 解决传统RAG中文本块缺乏上下文的问题
            llm_client: LLM客户端（支持OpenAI GPT或Claude）
                - 用于生成上下文化的文本块
                - 当前版本主要支持OpenAI GPT-3.5/GPT-4
                - 原Claude API代码已注释保留
            use_reranker: 是否使用重排序
                - 在初步检索结果基础上，使用专门的重排序模型优化结果排序
            rerank_function: 重排序函数（如Cohere Rerank）
        """
        self.collection_name = collection_name

        # 对于Milvus-lite，uri是本地路径，如"./milvus.db"
        # 对于Milvus独立服务，uri类似"http://localhost:19530"
        # 对于Zilliz Cloud，请设置`uri`和`token`
        self.client = MilvusClient(uri, token)

        self.embedding_function = dense_embedding_function

        self.use_sparse = use_sparse
        self.sparse_embedding_function = None

        self.use_contextualize_embedding = use_contextualize_embedding
        # self.anthropic_client = anthropic_client  # 注释掉Claude客户端
        self.llm_client = llm_client  # 改用通用LLM客户端

        self.use_reranker = use_reranker
        self.rerank_function = rerank_function

        # 参数验证：如果启用稀疏向量，必须提供稀疏嵌入函数
        if use_sparse is True and sparse_embedding_function:
            self.sparse_embedding_function = sparse_embedding_function
        elif use_sparse is True and sparse_embedding_function is None:
            raise ValueError(
                "稀疏嵌入函数不能为空，如果use_sparse为True"
            )
        else:
            pass

    def build_collection(self):
        """
        构建Milvus集合
        
        🏗️ 集合设计原理:
        1. Schema设计：定义数据结构和字段类型
        2. 索引策略：选择合适的索引类型以优化搜索性能
        3. 动态字段：支持灵活的元数据存储
        4. 向量字段：支持密集和稀疏向量的混合存储
        
        📊 存储结构:
        ┌─────────────┬──────────────┬─────────────────┬──────────────┐
        │    字段     │    类型      │      用途       │    索引类型   │
        ├─────────────┼──────────────┼─────────────────┼──────────────┤
        │ pk          │ INT64        │ 主键ID         │ 自动          │
        │ dense_vector│ FLOAT_VECTOR │ 语义向量       │ FLAT/IP      │
        │ sparse_vector│ SPARSE_VECTOR│ 关键词向量     │ INVERTED/IP  │
        │ content     │ VARCHAR      │ 原始内容       │ 动态字段     │
        │ metadata    │ JSON         │ 元数据信息     │ 动态字段     │
        └─────────────┴──────────────┴─────────────────┴──────────────┘
        
        集合设计说明：
        1. 使用动态Schema，支持灵活的字段添加
        2. 主键自动生成，确保每条记录的唯一性
        3. 密集向量字段：存储文本的语义向量表示
        4. 稀疏向量字段（可选）：存储关键词相关的稀疏向量
        5. 索引策略：密集向量使用FLAT索引，稀疏向量使用倒排索引
        """
        # 创建集合的Schema定义
        schema = self.client.create_schema(
            auto_id=True,  # 自动生成主键ID
            enable_dynamic_field=True,  # 允许动态添加字段，提高灵活性
        )
        
        # 添加主键字段
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        
        # 添加密集向量字段 - 存储文本的语义向量表示
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedding_function.dim,  # 向量维度由嵌入函数决定
        )
        
        # 如果启用稀疏向量，添加稀疏向量字段
        if self.use_sparse is True:
            schema.add_field(
                field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
            )

        # 准备索引参数 - 索引用于加速向量搜索
        index_params = self.client.prepare_index_params()
        
        # 为密集向量添加索引
        # FLAT索引：精确搜索，适合小到中等规模数据集
        # IP（内积）距离：适合归一化后的向量
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="IP"
        )
        
        # 为稀疏向量添加倒排索引
        if self.use_sparse is True:
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",  # 稀疏向量专用的倒排索引
                metric_type="IP",
            )

        # 创建集合
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            enable_dynamic_field=True,
        )

    def insert_data(self, chunk, metadata):
        """
        插入标准数据到Milvus
        
        📥 标准数据插入流程:
        这是基础的数据插入方法，实现最简单的文本块存储策略
        
        🔄 处理流程:
        1. 文本输入验证
        2. 密集向量生成（语义表示）
        3. 稀疏向量生成（可选，关键词表示）
        4. 数据结构组装
        5. Milvus批量插入
        
        📊 数据流向:
        原始文本块 → Embedding模型 → 向量表示 → Milvus存储
                                   ↓
        元数据信息 → 字段映射 → 动态字段存储
        
        参数:
            chunk: 文本块内容（原始文本，未经过上下文增强）
            metadata: 元数据（包含文档ID、块ID、原始内容等信息）
        
        存储内容：
        1. 密集向量：文本块的语义向量表示
        2. 稀疏向量（可选）：文本块的关键词向量表示
        3. 元数据：文档和块的标识信息
        """
        # 生成文本块的密集向量表示
        dense_vec = self.embedding_function([chunk])[0]
        
        # 构建要插入的数据
        data = {
            "dense_vector": dense_vec,
            **metadata  # 展开元数据字段
        }
        
        # 如果启用稀疏向量，生成并添加稀疏向量
        if self.use_sparse is True:
            sparse_vec = self.sparse_embedding_function([chunk])[0]
            data["sparse_vector"] = sparse_vec
            
        # 插入数据到Milvus集合
        self.client.insert(
            collection_name=self.collection_name,
            data=[data]
        )
        self.client.flush(collection_name=self.collection_name)

    def insert_contextualized_data(self, doc_content, chunk_content, metadata):
        """
        插入上下文化的数据
        
        🧠 上下文增强核心流程:
        这是实现上下文检索的关键方法，解决传统RAG的语义隔离问题
        
                 🔄 上下文化处理流程:
         1. 文档上下文准备
         2. LLM提示词构建
         3. OpenAI GPT API调用（上下文增强）
         4. 增强文本向量化
         5. 向量数据存储
        
                 📊 数据增强流程:
         原始文档 ──┐
                   ├─→ LLM提示词 → OpenAI GPT API → 上下文增强文本
         文本块 ───┘                                  ↓
                                             Embedding → 增强向量 → Milvus
        
        🎯 核心创新点:
        这是上下文检索的核心功能：
        1. 使用整个文档内容作为上下文
        2. 通过LLM（OpenAI GPT）丰富单个文本块的语义信息
        3. 使用增强后的文本生成向量并存储
        
        ✨ 上下文化的优势：
        - 解决文本块孤立的问题
        - 保留跨块的语义连贯性
        - 提高检索的准确性，特别是对于需要上下文理解的查询
        - 减少语义歧义和误解
        
        参数:
            doc_content: 整个文档内容（作为上下文背景）
            chunk_content: 当前文本块内容（需要被增强的部分）
            metadata: 元数据
        """
        # 构建LLM提示词，要求为文本块添加文档上下文
        prompt = f"""
        <文档>
        {doc_content}
        </文档>
        <块>
        {chunk_content}
        </块>
        
        我需要你对上述<块>进行丰富，使用<文档>中的内容提供背景和上下文信息。
        你的回答应该包含<块>的完整内容，并确保语义连贯。只返回丰富后的文本内容，不要添加任何说明或解释。
        
        目标：
        1. 保持原始块的核心信息不变
        2. 添加必要的背景上下文，使块的含义更加清晰
        3. 确保增强后的文本在语义上是连贯和完整的
        """
        
        # === OpenAI GPT API调用（新版本） ===
        # 调用OpenAI GPT API生成上下文化的文本块
        response = self.llm_client.chat.completions.create(
            # model="gpt-3.5-turbo",  # 使用GPT-3.5-turbo，经济高效
            # model="gpt-4",        # 可选择GPT-4获得更好效果
            model="deepseek-ai/DeepSeek-V3",
            max_tokens=1000,        # 限制输出长度
            temperature=0,          # 确保输出的一致性和可重复性
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # 提取OpenAI生成的上下文化文本
        contextualized_chunk = response.choices[0].message.content.strip()
        
        # === Claude API调用（原版本，已注释） ===
        # # 调用Claude API生成上下文化的文本块
        # message = self.anthropic_client.messages.create( 
        #     model="claude-3-haiku-20240307",  # 使用快速、经济的Haiku模型
        #     max_tokens=1000,  # 限制输出长度
        #     temperature=0,    # 确保输出的一致性和可重复性
        #     messages=[
        #         {"role": "user", "content": prompt}
        #     ]
        # )
        # 
        # # 提取LLM生成的上下文化文本
        # contextualized_chunk = message.content[0].text.strip()
        
        # 使用上下文化后的内容生成嵌入向量并插入
        dense_vec = self.embedding_function([contextualized_chunk])[0]
        data = {
            "dense_vector": dense_vec,
            "contextualized_content": contextualized_chunk,  # 保存增强后的内容
            **metadata
        }
        
        # 如果启用稀疏向量，也使用上下文化内容生成稀疏向量
        if self.use_sparse is True:
            sparse_vec = self.sparse_embedding_function([contextualized_chunk])[0]
            data["sparse_vector"] = sparse_vec
            
        # 插入上下文化数据到Milvus
        self.client.insert(
            collection_name=self.collection_name,
            data=[data]
        )

    def search(self, query, k=5):
        """
        搜索相关内容
        
        🔍 智能检索执行流程:
        这是检索系统的核心入口，支持多种检索策略的统一调用
        
        🎯 检索策略选择:
        ┌─────────────────┬──────────────────┬──────────────────┐
        │    检索阶段     │      处理方式    │      输出结果    │
        ├─────────────────┼──────────────────┼──────────────────┤
        │ 1. 查询向量化   │ Embedding模型    │ 查询向量表示     │
        │ 2. 相似度搜索   │ Milvus向量搜索   │ Top-K候选结果    │
        │ 3. 结果重排序   │ Cohere Rerank    │ 优化排序结果     │
        └─────────────────┴──────────────────┴──────────────────┘
        
        🔄 详细搜索流程：
        1. 查询预处理与向量化
        2. Milvus向量相似度搜索
        3. 初步结果获取与过滤
        4. 可选重排序优化
        5. 结果后处理与返回
        
        📊 搜索优化策略:
        查询文本 → 向量化 → 相似度搜索 → 候选结果
                                            ↓
        最终结果 ← 重排序优化 ← 语义匹配 ← 结果集
        
        参数:
            query: 查询文本
            k: 返回结果数量
        
        返回:
            搜索结果列表，按相关性排序
        """
        # 设置搜索参数
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        
        # 生成查询的嵌入向量
        dense_vec = self.embedding_function([query])[0]
        
        # 执行标准密集向量搜索
        # 这里使用内积（IP）作为相似度度量
        res = self.client.search(
            collection_name=self.collection_name,
            data=[dense_vec],
            limit=k,
            output_fields=["content", "contextualized_content"],  # 返回原始内容和上下文化内容
            search_params=search_params,
        )
        
        # 使用重排序器进一步优化结果
        # 重排序的作用：基于查询和文档的深层语义关系重新排序结果
        if self.use_reranker:
            # 提取文档内容用于重排序
            docs = []
            for hit in res[0]:
                # 优先使用上下文化内容（如果存在），否则使用原始内容
                content = hit["entity"].get("contextualized_content", hit["entity"].get("content", ""))
                docs.append(content)
            
            # 应用重排序：计算查询与每个文档的深层相关性分数
            rerank_results = self.rerank_function(query, docs)
            
            # 根据重排序结果重新排序原始结果
            reranked_results = []
            for result in rerank_results:
                idx = result.index  # 使用 .index 属性获取原始索引
                reranked_results.append(res[0][idx])
            
            res = [reranked_results]
        
        return res


def evaluate_retrieval(eval_data, retrieval_function, db, k=5):
    """
    评估检索性能 - 深度评估系统的核心函数
    
    📊 深度评估系统设计:
    这是一个全面的检索性能评估框架，采用多维度评估策略
    
    🔬 深度评估（Deep Evaluation）的概念：
    1. 不仅仅评估检索的数量，更关注检索的质量
    2. 使用多维度指标评估检索系统性能
    3. 通过黄金标准数据集进行客观评估
    4. 支持不同检索策略的公平对比
    
    🎯 评估流程设计:
    ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
    │   评估阶段      │    输入数据     │    处理方式     │    输出结果     │
    ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
    │ 1. 数据准备     │ 评估查询集      │ 格式验证       │ 标准化数据     │
    │ 2. 检索执行     │ 单个查询       │ 检索函数调用    │ 候选结果列表   │
    │ 3. 结果匹配     │ 检索结果       │ 精确文本匹配    │ 匹配成功计数   │
    │ 4. 性能计算     │ 匹配统计       │ 指标计算       │ 评估分数      │
    └─────────────────┴─────────────────┴─────────────────┴─────────────────┘
    
    📈 评估指标说明：
    - Pass@K: 在前K个检索结果中包含正确答案的查询比例
      * 计算公式: (包含正确答案的查询数 / 总查询数) × 100%
      * 反映系统的整体检索成功率
    - 平均分数: 每个查询检索到的正确文档块占总正确块的比例
      * 计算公式: Σ(查询检索到的正确块数 / 查询的总正确块数) / 总查询数
      * 反映系统的细粒度检索精度
    - 召回率: 评估系统找到相关文档的能力
      * 衡量系统在不遗漏重要信息方面的表现
    
    🔄 评估执行流程:
    评估数据 → 查询解析 → 黄金标准提取 → 检索执行 → 结果对比 → 指标计算
        ↓
    性能报告 ← 统计分析 ← 分数汇总 ← 匹配验证 ← 精确匹配
    
    参数:
        eval_data: 评估数据集
            - 包含查询和对应的黄金标准答案
            - 每个查询都有明确的正确文档块标识
        retrieval_function: 检索函数
            - 接受查询和数据库，返回检索结果
        db: 数据库实例
        k: 评估的top-k结果数
        
    返回:
        评估结果字典，包含Pass@K分数、平均分数等指标
    """
    total_score = 0      # 累计分数
    total_queries = 0    # 总查询数
    
    # 遍历每个评估查询
    for item in tqdm(eval_data, desc="Evaluating retrieval"):
        total_queries += 1
        query = item["query"]
        
        # 获取当前查询的黄金标准内容（正确答案）
        golden_contents = []
        for ref in item["references"]:
            doc_uuid = ref["doc_uuid"]      # 文档的唯一标识
            chunk_index = ref["chunk_index"] # 文档块的索引
            
            # 在数据集中查找对应的原始文档
            golden_doc = next(
                (
                    doc
                    for doc in dataset
                    if doc.get("original_uuid") == doc_uuid
                ),
                None,
            )
            if not golden_doc:
                print(f"警告：未找到UUID为{doc_uuid}的黄金文档")
                continue
                
            # 在文档中查找对应的文档块
            golden_chunk = next(
                (
                    chunk
                    for chunk in golden_doc["chunks"]
                    if chunk["original_index"] == chunk_index
                ),
                None,
            )
            if not golden_chunk:
                print(f"警告：在文档{doc_uuid}中未找到索引为{chunk_index}的黄金块")
                continue
                
            golden_contents.append(golden_chunk["content"].strip())
            
        if not golden_contents:
            print(f"警告：未找到查询的黄金内容：{query}")
            continue
            
        # 使用检索函数获取检索结果
        retrieved_docs = retrieval_function(query, db, k=k)
        
        # 计算有多少黄金块在top-k检索文档中被找到
        # 这是评估检索准确性的核心逻辑
        chunks_found = 0
        for golden_content in golden_contents:
            for doc in retrieved_docs[0][:k]:  # 只检查前k个结果
                content_field = "content"
                if "contextualized_content" in doc["entity"]:
                    # 评估时使用原始内容进行比较，确保公平性
                    content_field = "content"
                retrieved_content = doc["entity"][content_field].strip()
                
                # 精确匹配检查
                if retrieved_content == golden_content:
                    chunks_found += 1
                    break  # 找到匹配就跳出内层循环
                    
        # 计算当前查询的分数：找到的正确块数 / 总的正确块数
        query_score = chunks_found / len(golden_contents)
        total_score += query_score
        
    # 计算整体评估指标
    average_score = total_score / total_queries
    pass_at_n = average_score * 100  # 转换为百分比
    
    return {
        "pass_at_n": pass_at_n,           # Pass@K分数（百分比）
        "average_score": average_score,    # 平均分数（0-1之间）
        "total_queries": total_queries,    # 总查询数
    }


def retrieve_base(query: str, db, k: int = 20) -> List[Dict[str, Any]]:
    """
    基础检索函数
    
    这是一个简单的包装函数，调用数据库的搜索方法
    用于评估系统中的统一接口
    """
    return db.search(query, k=k)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    加载JSONL文件并返回字典列表
    
    JSONL格式：每行一个JSON对象，适合存储结构化的评估数据
    """
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def evaluate_db(db, original_jsonl_path: str, k):
    """
    评估数据库的检索性能
    
    这是评估流程的主入口函数：
    1. 加载评估数据集
    2. 运行检索评估
    3. 输出性能指标
    
    参数:
        db: 要评估的数据库实例
        original_jsonl_path: 评估数据集文件路径
        k: 评估的top-k参数
    
    返回:
        评估结果字典
    """
    # 加载原始JSONL数据作为查询和真实标签
    original_data = load_jsonl(original_jsonl_path)
    
    # 评估检索性能
    results = evaluate_retrieval(original_data, retrieve_base, db, k)
    
    # 输出评估结果
    print(f"Pass@{k}: {results['pass_at_n']:.2f}%")
    print(f"总分: {results['average_score']}")
    print(f"总查询数: {results['total_queries']}")
    
    return results


def download_data():
    """
    下载示例数据
    
    从Anthropic的GitHub仓库下载演示数据：
    1. codebase_chunks.json: 代码库的文档块数据
    2. evaluation_set.jsonl: 评估查询和标准答案
    """
    import urllib.request
    
    # 检查文件是否已存在，避免重复下载
    if not os.path.exists("codebase_chunks.json"):
        print("下载codebase_chunks.json...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/anthropics/anthropic-cookbook/refs/heads/main/skills/contextual-embeddings/data/codebase_chunks.json",
            "codebase_chunks.json"
        )
    
    if not os.path.exists("evaluation_set.jsonl"):
        print("下载evaluation_set.jsonl...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/anthropics/anthropic-cookbook/refs/heads/main/skills/contextual-embeddings/data/evaluation_set.jsonl",
            "evaluation_set.jsonl"
        )
    
    print("数据下载完成！")


def main():
    """
    主函数 - 运行所有实验
    
    🧪 实验设计框架:
    这是一个完整的对比实验系统，采用控制变量法验证不同检索策略的效果
    
    🎯 实验目标:
    - 验证上下文检索相对于标准检索的性能提升
    - 量化重排序技术的额外收益
    - 建立检索技术的性能基准
    - 为实际应用提供技术选择指导
    
    📊 实验设计矩阵:
    ┌─────────────────┬────────────┬────────────┬────────────┬──────────────┐
    │     实验组      │ 文本预处理 │ 向量检索   │ 结果重排序 │  预期性能    │
    ├─────────────────┼────────────┼────────────┼────────────┼──────────────┤
    │ 标准检索(基线)  │ 原始文本块 │ 密集向量   │ 无        │ 基准性能     │
    │ 上下文检索      │ LLM增强块  │ 密集向量   │ 无        │ 中等提升     │
    │ 重排序检索      │ LLM增强块  │ 密集向量   │ Cohere    │ 最佳性能     │
    └─────────────────┴────────────┴────────────┴────────────┴──────────────┘
    
    🔬 实验控制变量:
    - 相同的数据集（codebase_chunks.json）
    - 相同的评估查询（evaluation_set.jsonl）
    - 相同的嵌入模型（BGE-large-zh）
    - 相同的评估指标（Pass@5）
    - 相同的向量数据库配置
    
    🚀 实验执行流程:
    1️⃣ 环境准备阶段:
       - API密钥配置验证
       - 模型初始化和测试
       - 数据下载和验证
    
    2️⃣ 数据预处理阶段:
       - 文档数据加载和解析
       - 评估查询集构建
       - 数据格式标准化
    
    3️⃣ 实验执行阶段:
       - 标准检索实验（基线）
       - 上下文检索实验（核心创新）
       - 重排序检索实验（性能优化）
    
    4️⃣ 结果分析阶段:
       - 性能指标统计
       - 改进幅度计算
       - 实验结论总结
    
    🎯 核心实验假设:
    这个函数运行三个对比实验，展示不同检索策略的性能差异：
    
    1. 标准检索：基线方法，直接使用原始文本块
    2. 上下文检索：使用LLM增强文本块的上下文信息
    3. 带重排序的上下文检索：在上下文检索基础上加入重排序优化
    
    ⚖️ 实验公平性保证:
    每个实验都使用相同的评估数据集，确保比较的公平性
    """
    # 替换这些为你的实际API密钥
    # cohere_api_key = os.getenv("COHERE_API_KEY")      # Cohere重排序API密钥
    # anthropic_api_key = os.getenv("CLAUDE_API_KEY")   # Claude API密钥（已注释）
    # openai_api_key = os.getenv("OPENAI_API_KEY")      # OpenAI API密钥（新增）
    
    # 下载示例数据
    download_data()
    
    # 加载数据集
    global dataset
    with open("codebase_chunks.json", "r") as f:
        dataset = json.load(f)
    
    # 只使用前5个文档进行测试（减少API调用成本和运行时间）
    dataset = dataset[:5]
    
    # 初始化各种模型和函数
    dense_ef = SentenceTransformerEmbeddingFunction(model_name='BAAI/bge-large-zh')  # 使用中文优化的BGE模型
    # cohere_rf = CohereRerankFunction(api_key=cohere_api_key)  # Cohere重排序函数
    bge_rf = BGERerankFunction(
        api_key=os.getenv("SILICON_FLOW_API_KEY"),
        base_url=os.getenv("SILICON_FLOW_BASE_URL"),
    )  # BGE重排序函数，默认使用模型BAAI/bge-reranker-v2-m3，模型下载到本地huggingface的缓存目录下
    
    # === OpenAI客户端初始化（新版本） ===
    # openai_client = openai.OpenAI(api_key=openai_api_key)  # OpenAI客户端
    deepseek_client = openai.OpenAI(
        api_key=os.getenv("SILICON_FLOW_API_KEY"),
        base_url=os.getenv("SILICON_FLOW_BASE_URL"),
    )  # DeepSeek客户端
    
    # === Claude客户端初始化（原版本，已注释） ===
    # anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)  # Claude客户端
    
    # ===============================
    # 实验一：标准检索（基线方法）
    # ===============================
    print("\n===== 实验一：标准检索 =====")
    print("这是基线实验，使用原始文本块进行检索，没有任何增强")
    
    standard_retriever = MilvusContextualRetriever(
        uri="http://172.17.19.130:19530", 
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="contextual_retrieval_baseline", 
        dense_embedding_function=dense_ef
    )
    
    # 构建集合并插入标准数据
    standard_retriever.build_collection()
    for doc in tqdm(dataset, desc="插入标准检索数据"):
        doc_content = doc["content"]
        for chunk in doc["chunks"]:
            metadata = {
                "doc_id": doc["doc_id"],
                "original_uuid": doc["original_uuid"],
                "chunk_id": chunk["chunk_id"],
                "original_index": chunk["original_index"],
                "content": chunk["content"],
            }
            chunk_content = chunk["content"]
            standard_retriever.insert_data(chunk_content, metadata)
    
    # 创建简化的评估数据（用于演示）
    # 在实际应用中，应该使用专门设计的评估数据集
    eval_data = []
    for doc in dataset[:2]:  # 只使用前2个文档进行评估
        for chunk in doc["chunks"][:2]:  # 每个文档只取前2个块
            eval_data.append({
                "query": chunk["content"][:50],  # 使用块内容的前50个字符作为查询
                "references": [{
                    "doc_uuid": doc["original_uuid"],
                    "chunk_index": chunk["original_index"]
                }]
            })
    
    # 保存评估数据
    with open("evaluation_set.jsonl", "w") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # 评估标准检索性能
    standard_results = evaluate_db(standard_retriever, "evaluation_set.jsonl", 5)
    
    # ===============================
    # 实验二：上下文检索
    # ===============================
    print("\n===== 实验二：上下文检索 =====")
    print("使用OpenAI GPT为每个文本块添加文档上下文，解决语义隔离问题")
    
    contextual_retriever = MilvusContextualRetriever(
        uri="http://172.17.19.130:19530", 
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="contextual_retrieval",
        dense_embedding_function=dense_ef,
        use_contextualize_embedding=True,  # 启用上下文化
        llm_client=deepseek_client,  # 使用OpenAI客户端
        # anthropic_client=anthropic_client,  # 原Claude客户端（已注释）
    )
    
    # 构建集合并插入上下文化数据
    contextual_retriever.build_collection()
    for doc in tqdm(dataset, desc="插入上下文检索数据"):
        doc_content = doc["content"]
        for chunk in doc["chunks"]:
            metadata = {
                "doc_id": doc["doc_id"],
                "original_uuid": doc["original_uuid"],
                "chunk_id": chunk["chunk_id"],
                "original_index": chunk["original_index"],
                "content": chunk["content"],
            }
            chunk_content = chunk["content"]
            # 使用上下文化插入方法
            contextual_retriever.insert_contextualized_data(
                doc_content, chunk_content, metadata
            )
    
    # 评估上下文检索性能
    contextual_results = evaluate_db(contextual_retriever, "evaluation_set.jsonl", 5)
    
    # ===============================
    # 实验三：带重排序的上下文检索
    # ===============================
    print("\n===== 实验三：带重排序的上下文检索 =====")
    print("在上下文检索基础上，使用Cohere重排序模型进一步优化结果")
    
    # 启用重排序功能
    contextual_retriever.use_reranker = True
    contextual_retriever.rerank_function = bge_rf
    
    # 评估带重排序的检索性能
    reranker_results = evaluate_db(contextual_retriever, "evaluation_set.jsonl", 5)
    
    # ===============================
    # 结果对比分析
    # ===============================
    print("\n===== 所有实验结果比较 =====")
    print("性能提升分析：")
    print(f"标准检索 Pass@5: {standard_results['pass_at_n']:.2f}%")
    print(f"上下文检索 Pass@5: {contextual_results['pass_at_n']:.2f}%")
    print(f"带重排序的上下文检索 Pass@5: {reranker_results['pass_at_n']:.2f}%")
    
    # 计算改进幅度
    context_improvement = contextual_results['pass_at_n'] - standard_results['pass_at_n']
    rerank_improvement = reranker_results['pass_at_n'] - standard_results['pass_at_n']
    
    print(f"\n性能改进分析：")
    print(f"上下文检索相比标准检索提升: {context_improvement:.2f}个百分点")
    print(f"重排序进一步提升: {rerank_improvement:.2f}个百分点")
    print(f"总体提升: {rerank_improvement:.2f}个百分点")


if __name__ == "__main__":
    main()
