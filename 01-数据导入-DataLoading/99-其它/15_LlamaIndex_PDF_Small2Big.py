from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.node_parser import SentenceSplitter

from dotenv import load_dotenv
load_dotenv()   

import os

# Setup LLM and embedding models
model_provider_api_key = os.getenv("O3_API_KEY")
model_provider_url_base = os.getenv("O3_BASE_URL")
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=model_provider_api_key,
    api_base=model_provider_url_base
)
llm = OpenAI(
    model="gpt-4o",
    api_key=model_provider_api_key,
    api_base=model_provider_url_base
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=72, chunk_overlap=20)

# Create sentence window parser
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
# Note:
# SentenceWindowNodeParser接收SentenceSplitter的结果，为每个句子创建各包含前后
# 3个句子作为上下文，双重存储扩展后上下文的metadata和原始句子的metadata。在检索阶
# 段依然使用原始句子做精准搜索，但给LLM提供检索结果时，提供扩展后带上下文的语料，既
# 保证检索的精确，又保留丰富的上下文信息。劣势是计算开销显著增大，对标题摘要类短文本、
# JSON等结构化数据不适用

# Load PDF and parse into nodes with sentence windows
loader = PDFReader()
documents = loader.load_data(file="90-文档-Data/复杂PDF/uber_10q_march_2022.pdf")
nodes = node_parser.get_nodes_from_documents(documents)

# Create index from nodes
index = VectorStoreIndex(nodes)

# Create query engine with metadata replacement
query_engine = index.as_query_engine(
    similarity_top_k=3,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
    verbose=True
)

# Example queries
query = "What is the change of free cash flow and what is the rate from the financial and operational highlights?"
# query = "how much COVID-19 response initiatives in millions in year 2021?" # 这个问题LC会LI不会
# query = "What is the Adjusted EBITDA loss in year COVID-19?"
# query = "how much is the Loss from operations for the period ended March 31, 2021?"
# query = "how much is the Loss from operations for 2022?"

response = query_engine.query(query)
print("\n************LlamaIndex Query Response************")
print(response)

# Display retrieved chunks
print("\n************Retrieved Text Chunks************")
for i, source_node in enumerate(response.source_nodes):
    print(f"\nChunk {i+1}:")
    print("Original sentence:")
    print(source_node.node.metadata["original_text"])
    print("\nContext window:")
    print(source_node.node.metadata["window"])
    print("-" * 50)