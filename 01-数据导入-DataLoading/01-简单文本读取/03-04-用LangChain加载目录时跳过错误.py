from langchain_community.document_loaders import DirectoryLoader, TextLoader
# 加载目录下所有文件，跳过出错文件，因为有些文件是图片，TextLoader 无法加载
import os
# 获取当前脚本文件所在的目录
script_dir = os.path.dirname(__file__)
print(f"获取当前脚本文件所在的目录：{script_dir}") 
# 结合相对路径构建完整路径
data_dir = os.path.join(script_dir, '../../90-文档-Data/黑悟空')

# 加载目录下所有 Markdown 文件
loader = DirectoryLoader(data_dir,
                          silent_errors=True, # Do not work: Error loading file /mnt/c/Workspace/github/rag-in-action-forked/01-数据导入-DataLoading/01-简单文本读取/../../90-文档-Data/黑悟空/黑悟空英文.jpg
                         loader_cls=TextLoader
                         )

docs = loader.load()
print(docs[0].page_content[:100])  # 打印第一个文档内容的前100个字符
