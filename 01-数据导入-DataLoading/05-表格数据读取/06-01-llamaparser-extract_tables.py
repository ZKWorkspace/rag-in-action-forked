from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_parse import LlamaParse
import time
import os
from dotenv import load_dotenv

# 加载环境变量（确保有OpenAI API密钥）
load_dotenv()
model_provider_api_key = os.getenv("O3_API_KEY")
model_provider_url_base = os.getenv("O3_BASE_URL")
# 设置基础模型
Settings.llm = OpenAI(model="gpt-4o",
                      api_key=model_provider_api_key,
                      api_base=model_provider_url_base)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small",
                                       api_key=model_provider_api_key,
                                       api_base=model_provider_url_base)

# 定义PDF路径
pdf_path = "90-文档-Data/复杂PDF/billionaires_page-1-5.pdf"

# 记录开始时间
start_time = time.time()

# 使用LlamaParse解析PDF
documents = LlamaParse(result_type="markdown").load_data(pdf_path)

# 记录结束时间
end_time = time.time()
print(f"PDF解析耗时: {end_time - start_time:.2f}秒")

# 打印解析结果
print("\n解析后的文档内容:")
for i, doc in enumerate(documents, 1):
    print(f"\n文档 {i} 内容:")
    print(doc.text)
