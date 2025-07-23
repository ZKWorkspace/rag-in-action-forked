from langchain_core.output_parsers import JsonOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# 定义输出格式
parser = JsonOutputParser()
prompt = PromptTemplate.from_template("请返回JSON格式的用户信息：{query}")
# 调用大模型并解析
llm = ChatDeepSeek(
    model="deepseek-ai/DeepSeek-V3",
    api_base=os.getenv("SILICON_FLOW_BASE_URL"),
    api_key=os.getenv("SILICON_FLOW_API_KEY"),
)
output = llm.invoke(prompt.format(query="用户ID 123"))
# 从 AIMessage 中提取内容
parsed_output = parser.parse(output.content)
print(parsed_output)

chain = prompt | llm | parser
print(chain.invoke({"query": "用户ID 456"}))
