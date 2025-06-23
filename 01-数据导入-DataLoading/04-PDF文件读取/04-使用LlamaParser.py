# 需要LLAMA_CLOUD_API_KEY
import os
from dotenv import load_dotenv
load_dotenv()   

# LlamaParse PDF reader for PDF Parsing
from llama_parse import LlamaParse
documents = LlamaParse(result_type="markdown",
                       api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                       show_progress=True,
                       verbose=True).load_data(
    "90-文档-Data/黑悟空/黑神话悟空.pdf"
)
print(documents)
print("-" * 100)

from llama_index.core.node_parser import MarkdownElementNodeParser
# Fix Error: No API key found for OpenAI
from llama_index.llms.openai import OpenAI
llm = OpenAI(
    model="gpt-4o",
    api_key=os.getenv("O3_API_KEY"),
    api_base=os.getenv("O3_BASE_URL"),
    # 如果你的 API 端点有其他需要传递的参数，可以在这里添加
    # 例如: temperature=0.7
)
node_parser = MarkdownElementNodeParser(llm=llm,
                                        # num_workers=4, # if multiple files passed, split in `num_workers` API calls
                                        # show_progress=True
                                        )
nodes = node_parser.get_nodes_from_documents(documents)

# print(nodes)
for node in nodes:
    print(node)
    print('-' * 100)

