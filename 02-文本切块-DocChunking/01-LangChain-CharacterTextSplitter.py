from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
loader = TextLoader("90-文档-Data/山西文旅/云冈石窟.txt")
documents = loader.load()
# 设置分块器，指定块的大小为50个字符，无重叠
text_splitter = CharacterTextSplitter(
    chunk_size=300,  # 每个文本块的大小为50个字符, 建议200~500
    chunk_overlap=50,  # 文本块之间没有重叠部分，建议是chunk_size的10%~20%
    # separator="\n\n" # 默认切块分隔符是'\n'，也支持自定义，但不像RecursiveCharacterTextSplitter，CharacterTextSplitter只支持单个分隔符
)

# Warning: Created a chunk of size 216, which is longer than the specified 100
# CharacterTextSplitter（以及 RecursiveCharacterTextSplitter）的分块逻辑是优先按
# 你指定的分隔符（如段落、句号、逗号、空格等）进行分割，只有在找不到合适分隔符时，才会
# 直接按字符数硬切。如果遇到一个很长的段落、句子或代码块，它本身就超过了 chunk_size，
# 而且中间没有你指定的分隔符，那么分词器会优先保证语义完整性，而不强制截断，那么这个块
# 就无法再细分，只能整体作为一个chunk。这时chunk的实际长度就会大于你设置的chunk_size，
# 并产生警告。
chunks = text_splitter.split_documents(documents)
print("\n=== 文档分块结果 ===")
for i, chunk in enumerate(chunks, 1):
    print(f"\n--- 第 {i} 个文档块 ---")
    print(f"内容: {chunk.page_content}")
    print(f"元数据: {chunk.metadata}")
    print("-" * 50)
