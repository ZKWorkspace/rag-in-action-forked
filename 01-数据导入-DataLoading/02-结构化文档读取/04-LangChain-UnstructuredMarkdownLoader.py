from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

markdown_path = "90-文档-Data/黑悟空/黑悟空版本介绍.md"
loader = UnstructuredMarkdownLoader(markdown_path)

data = loader.load()
print(data[0].page_content[:250]) # Note: len(data) == 1

# Note: File content is sliced into 22 parts 
# and relations between docs are built by 'element_id'
# and 'parent_id' metadata.
# Note: strategy can be one of 'hi_res', 'fast'
loader = UnstructuredMarkdownLoader(markdown_path, mode="elements", strategy="hi_res")
data = loader.load()
print(f"Number of documents: {len(data)}\n")
for document in data:
    print(f"{document}\n")
