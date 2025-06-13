import os
import time
import base64
from PIL import Image
from dotenv import load_dotenv
# Fix Error: pdf2image.exceptions.PDFInfoNotInstalledError: Unable to get page count. Is poppler installed and in PATH?
# pdf2image is only a wrapper around poppler, so you need intall poppler firstly.
# sudo apt-get update && sudo apt-get install poppler-utils
from pdf2image import convert_from_path

# from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 初始化 OpenAI 客户端
# client = OpenAI()
load_dotenv()
model_provider_api_key = os.getenv("GITEE_API_KEY")
model_provider_url_base = os.getenv("GITEE_URL_BASE")
print(f"Model provider api key : {model_provider_api_key}")
print(f"Model provider url base: {model_provider_url_base}")
client = ChatOpenAI(
    model="Qwen2-VL-72B",
    base_url=model_provider_url_base,
    api_key=model_provider_api_key,
)

# 创建提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个专业的图片分析助手。请分析图片中的内容，要求：\n"
                  "1. 首先用50字以内概括图片的主题和核心内容\n"
                  "2. 然后详细列举图片中的所有文字内容，包括：\n"
                  "   - 标题和副标题\n"
                  "   - 正文段落\n"
                  "   - 列表项\n"
                  "   - 图表说明\n"
                  "   - 其他文字信息\n"
                  "请按以下格式输出：\n"
                  "【主题概述】\n"
                  "（50字以内的主题概括）\n\n"
                  "【文字内容】\n"
                  "（按类别列举所有文字内容）\n\n"),
        ("human", "请分析以下图片的内容：\n{img_url}")
    ]
)

chain = prompt | client
output_dir = "temp_images"

# 1. PDF 转图片
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Fix Error: llama-3.2-90b-vision-instruct's maximum context length is 131072 tokens,
# However, default dpi(200) requests 369249 tokens in the messages.
images = convert_from_path("90-文档-Data/黑悟空/黑神话悟空.pdf", dpi=100)
image_paths = []
for i, image in enumerate(images):
    # 调整图片尺寸
    # 计算新的尺寸，保持宽高比
    max_size = (400, 400)  # 设置最大尺寸
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    image_path = os.path.join(output_dir, f'page_{i+1}.jpg')
    # 使用较低的JPEG质量来进一步减小文件大小
    image.save(image_path, 'JPEG', quality=85)
    image_paths.append(image_path)
print(f"成功转换 {len(image_paths)} 页")


# 2. GPT-4o 分析图片
print("\n开始分析图片...")
results = []
for image_path in image_paths:
    try:
        # 读取本地图片文件并转换为base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 构建图片URL（使用base64编码）
        img_url = f"data:image/jpeg;base64,{base64_image}"
        print(f"正在分析图片: {image_path}")
        
        # 调用模型分析图片
        response = chain.invoke({"img_url": img_url})
        results.append(response.content)
        print(f"图片分析完成: {image_path}")

        print("等待1分钟后继续...") # Fix Error: http error 429, Too Many Requests
        time.sleep(60)  # 60秒 = 1分钟
    except Exception as e:
        print(f"分析图片时出错: {str(e)}")
        continue


# 3. 转换为 LangChain 的 Document 数据结构
from langchain_core.documents import Document

documents = [
    Document(
        page_content=result,
        metadata={"source": "data/黑悟空/黑神话悟空.pdf", "page_number": i + 1}
    )
    for i, result in enumerate(results)
]

# 输出所有生成的 Document 对象
print("\n分析结果：")
for doc in documents:
    print(f"内容: {doc.page_content}\n元数据: {doc.metadata}\n")
    print("-" * 80)

# 清理临时文件
for image_path in image_paths:
    os.remove(image_path)
os.rmdir(output_dir)

