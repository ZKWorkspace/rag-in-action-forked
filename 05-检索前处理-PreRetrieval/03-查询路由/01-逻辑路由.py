import os
from dotenv import load_dotenv
load_dotenv()
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field # Fix Warning: As of langchain-core 0.3.0, LangChain uses pydantic v2 internally.
from pydantic import BaseModel, Field
from langchain_deepseek import ChatDeepSeek

# 数据模型
class RouteQuery(BaseModel):
    """将用户查询路由到最相关的数据源"""
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="根据用户问题，选择最适合回答问题的数据源",
    )

def create_router():
    """创建并返回路由模型"""
    # 带函数调用的大模型
    llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)
    
    # 提示模板
    system = """你是将用户问题路由到合适数据源的专家。
根据问题所涉及的编程语言，将其路由到相关的数据源。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])
    
    # 定义路由器
    return prompt | structured_llm

def route_question(question: str) -> str:
    """路由用户问题到合适的数据源"""
    router = create_router()
    result = router.invoke({"question": question})
    return result.datasource

# 使用示例
if __name__ == "__main__":
    # 测试问题
    test_question = "Python中的列表和元组有什么区别？"
    # test_question = "JavaScript的闭包概念"
    # test_question = "Go语言的并发模型"
    # test_question = "如何实现一个简单的HTTP服务器？"
    # test_question = "明天天气如何？" # Note:虽然是个毫不相干的询问，但是依然会结合上下文在datasource中选一个！
    result = route_question(test_question)
    print(f"问题: {test_question}")
    print(f"路由结果: {result}")

