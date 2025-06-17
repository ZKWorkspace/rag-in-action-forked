import os
import pdfplumber
import pandas as pd
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document, VectorStoreIndex, Settings
from typing import List

load_dotenv()
model_provider_api_key = os.getenv("O3_API_KEY")
model_provider_url_base = os.getenv("O3_URL_BASE")
Settings.llm  = OpenAI(model="gpt-4o", 
                       api_key=model_provider_api_key,
                       api_base=model_provider_url_base)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=model_provider_api_key,
    api_base=model_provider_url_base,
    # 有些 embedding 端点可能也接受额外参数
)

pdf_path = "90-文档-Data/复杂PDF/billionaires_page-1-5.pdf"

# 打开 PDF 并解析表格
with pdfplumber.open(pdf_path) as pdf:
    tables = {}
    for i, page in enumerate(pdf.pages, 1):
        tables[i] = []
        table = page.extract_table()
        if table:
            tables[i].append(table)

# 转换所有表格为 DataFrame 并构建文档
documents: List[Document] = []
for page_id in tables.keys():
    if tables[page_id]:
        # 遍历所有表格
        for i, table in enumerate(tables[page_id], 1):
            # 将表格转换为 DataFrame
            df = pd.DataFrame(table)
            
            # 保存到CSV文件
            # csv_filename = f"billionaires_table_{i}.csv"
            # df.to_csv(csv_filename, index=False)
            # print(f"\n表格 {i} 数据已保存到 {csv_filename}")
            
            # 将DataFrame转换为文本
            text = df.to_string()
            
            # 创建Document对象
            doc = Document(text=text, metadata={"page": f"页{page_id}", "source": f"表格{i}"})
            documents.append(doc)

# 构建索引
print(documents)
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
page = "3"
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="page", value=f"页{page}"),
    ]
)
query_engine = index.as_query_engine(filters=filters)

# 示例问答
questions = [
    "谁是最富有的人?",
    "最年轻的富豪是谁?"
]

print("\n===== 问答演示 =====")
for question in questions:
    response = query_engine.query(question)
    print(f"\n问题: {question}")
    print(f"回答: {response}")