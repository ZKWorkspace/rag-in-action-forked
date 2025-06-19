from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = TextLoader("90-文档-Data/山西文旅/云冈石窟.txt")
documents = loader.load()
# 定义分割符列表，按优先级依次使用
# 适合中文文本的分隔符
# separators = [
#     "\n\n",   # 段落
#     "\n",     # 换行
#     "。",     # 中文句号
#     "！",     # 中文感叹号
#     "？",     # 中文问号
#     "；",     # 中文分号
#     "，",     # 中文逗号
#     " ",      # 空格
#     ""        # 最后兜底，强制切分
# ]
# 适合英文文本的分隔符
# separators = [
#     "\n\n",   # 段落
#     "\n",     # 换行
#     ".",      # 句号
#     "!",      # 感叹号
#     "?",      # 问号
#     ";",      # 分号
#     ",",      # 逗号
#     " ",      # 空格
#     ""        # 最后兜底
# ]
# 适合中英文混合分隔符
# separators = [
#     "\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", ",", "，", " ", ""
# ]
separators = ["\n\n", "。", "，", " "] # . 是句号，， 是逗号， 是空格
# 创建递归分块器，并传入分割符列表
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    separators=separators
)
chunks = text_splitter.split_documents(documents)
print("\n=== 文档分块结果 ===")
for i, chunk in enumerate(chunks, 1):
    print(f"\n--- 第 {i} 个文档块 ---")
    print(f"内容: {chunk.page_content}")
    print(f"元数据: {chunk.metadata}")
    print("-" * 50)