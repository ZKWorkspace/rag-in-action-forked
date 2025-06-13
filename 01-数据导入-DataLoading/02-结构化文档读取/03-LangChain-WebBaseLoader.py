# 使用WebBaseLoader加载网页
import bs4
# Fix warning: USER_AGENT environment variable not set, consider setting it to identify your requests.
# Please set env variable USER_AGENT before WebBaseLoader importing
import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
from langchain_community.document_loaders import WebBaseLoader
# page_url = "https://zh.wikipedia.org/wiki/黑神话：悟空"
page_url = "https://www.chinanews.com.cn/gj/2025/06-11/10430400.shtml"
# loader = WebBaseLoader(web_paths=[page_url])
# docs = []
# docs = loader.load()
# assert len(docs) == 1
# doc = docs[0]
# print(f"{doc.metadata}\n")
# print(doc.page_content.strip()[:3000])


# 只解析文章的主体部分
loader = WebBaseLoader(
    web_paths=[page_url],
    # bs_kwargs={
    #     "parse_only": bs4.SoupStrainer(id="bodyContent"), # Note: 只解析id为bodyContent的元素
    # },
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(class_="content_maincontent_content"),
    },
    # bs_get_text_kwargs={"separator": " | ", "strip": True},
)
docs = []
docs = loader.load()
assert len(docs) == 1
doc = docs[0]
print(f"{doc.metadata}\n")
print(doc.page_content)
