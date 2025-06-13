from langchain_unstructured import UnstructuredLoader
from typing import List
from langchain_core.documents import Document
# page_url = "https://zh.wikipedia.org/wiki/黑神话：悟空"
page_url = "https://www.chinanews.com.cn/gj/2025/06-11/10430400.shtml"
def _get_setup_docs_from_url(url: str) -> List[Document]:
    loader = UnstructuredLoader(web_url=url)
    setup_docs = []
    # parent_id = None  # 初始化 parent_id
    current_parent = None  # 用于存储当前父元素
    for doc in loader.load():
        # 检查是否是 Title 或 Table
        if doc.metadata["category"] == "Title" or doc.metadata["category"] == "Table":
            # parent_id = doc.metadata["element_id"]
            current_parent = doc  # 更新当前父元素
            setup_docs.append(doc)
        elif current_parent is not None :
            setup_docs.append((current_parent, doc))  # 将父元素和子元素一起存储
        else:
            setup_docs.append(doc)
    return setup_docs       
docs = _get_setup_docs_from_url(page_url)
for item in docs:
    if isinstance(item, tuple):
        parent, child = item
        if parent is not None and child is not None:
            print(f'父元素 - {parent.metadata["category"]}: {parent.page_content}')
            print(f'子元素 - {child.metadata["category"]}: {child.page_content}')
        else:
            parent_status = 'invalid' if parent is None else 'valid'
            child_status = 'invalid' if child is None else 'valid'
            print(f"Whoops, parent is {parent_status:<8}, child is {child_status:<8}")
    else:
        print(f'{item.metadata["category"]}: {item.page_content}')
    print("-" * 80)

