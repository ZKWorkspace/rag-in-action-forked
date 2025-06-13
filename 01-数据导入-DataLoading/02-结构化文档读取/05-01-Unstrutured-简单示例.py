# 使用UnstructuredLoader加载网页
from langchain_unstructured import UnstructuredLoader # pip install langchain-unstructured
# page_url = "https://zh.wikipedia.org/wiki/黑神话：悟空"
page_url = "https://www.chinanews.com.cn/gj/2025/06-11/10430400.shtml"
loader = UnstructuredLoader(web_url=page_url)
docs = loader.load()
# for doc in docs[:5]:
for doc in docs:
    print(f'{doc.metadata["element_id"]}: {doc.metadata["category"]:<17}: {doc.page_content} ')
    # print(f'{doc.metadata["element_id"]}: {str(doc.metadata["parent_id"]) if hasattr(doc.metadata, 'parent_id') else "":<32}: {doc.metadata["category"]:<17}: {doc.page_content} ')

