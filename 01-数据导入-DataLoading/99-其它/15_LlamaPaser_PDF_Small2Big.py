from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

from dotenv import load_dotenv
load_dotenv()   

# Setup LLM and embedding models
import os
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'

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

# Create sentence window parser
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# Load PDF and parse into nodes with sentence windows
# LlamaParse PDF reader for PDF Parsing
from llama_parse import LlamaParse
parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"), result_type='markdown')
documents = parser.load_data("90-文档-Data/复杂PDF/uber_10q_march_2022.pdf")
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
# query = "how much COVID-19 response initiatives in millions in year 2021?"
# query = "After the year of COVID-19, how much EBITDA profit improved?"

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