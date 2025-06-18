import os
from dotenv import load_dotenv
import pymupdf4llm # pip install pymupdf4llm
import pymupdf.pro # pip install pymupdfpro
load_dotenv()
pymupdf.pro.unlock(os.getenv("PYMUPDF_PRO_TRIAL_KEY"))

# 打开PDF文件
llama_reader = pymupdf4llm.LlamaMarkdownReader() # 效果不理想
llama_docs = llama_reader.load_data("90-文档-Data/山西文旅/壶口瀑布-ch.pdf")
print(llama_docs) # 输出llama_index的Document类型实例