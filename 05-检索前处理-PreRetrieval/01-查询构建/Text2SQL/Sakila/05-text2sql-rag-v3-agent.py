# text2sql_query.py
import os
import logging
import yaml
import openai
import re
from dotenv import load_dotenv
from pymilvus import MilvusClient, MilvusException
from pymilvus import model
from sqlalchemy import create_engine, text

# 1. 环境与日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()  # 加载 .env 环境变量

# 2. 初始化 OpenAI API（使用最新 Response API）
openai.api_key = os.getenv("DEEPSEEK_API_KEY")
openai.base_url = os.getenv("DEEPSEEK_BASE_URL")

# 建议使用新 Response API 风格
# 例如: openai.chat.completions.create(...) 而非旧的 ChatCompletion.create

MODEL_NAME = os.getenv("OPENAI_MODEL", "deepseek-chat")

# 3. 嵌入函数初始化
def init_embedding():
    return model.dense.OpenAIEmbeddingFunction(
        model_name='BAAI/bge-m3',
        base_url=os.getenv("SILICON_FLOW_BASE_URL"),
        api_key=os.getenv("SILICON_FLOW_API_KEY")
    )

# 4. Milvus 客户端连接
client = MilvusClient(uri="http://172.17.19.130:19530", token=os.getenv("MILVUS_TOKEN"))
DATABASE_NAME = "sakila"
if DATABASE_NAME not in client.list_databases():
    try:
        client.create_database(db_name=DATABASE_NAME)
        logging.info(f"✓ {DATABASE_NAME} 创建成功")
    except MilvusException as e:
        logging.error(str(e))
        exit
client.use_database(db_name=DATABASE_NAME)

# 5. 嵌入函数实例化
embedding_fn = init_embedding()

# 6. 数据库连接（SAKILA）
DB_URL = os.getenv(
    "SAKILA_DB_URL", 
    f"mysql+pymysql://{os.getenv("MYSQL_USR")}:{os.getenv("MYSQL_PWD")}@{os.getenv("MYSQL_HOST")}:{os.getenv("MYSQL_PORT")}/sakila"
)
engine = create_engine(DB_URL)

# 7. 检索函数
def retrieve(collection: str, query_emb: list, top_k: int = 3, fields: list = None):
    results = client.search(
        collection_name=collection,
        data=[query_emb],
        limit=top_k,
        output_fields=fields
    )
    logging.info(f"[检索] {collection} 检索结果: {results}")
    return results[0]  # 返回第一个查询的结果列表

# 8. SQL 提取函数
def extract_sql(text: str) -> str:
    # 尝试匹配 SQL 代码块
    sql_blocks = re.findall(r'```sql\n(.*?)\n```', text, re.DOTALL)
    if sql_blocks:
        return sql_blocks[0].strip()
    
    # 如果没有找到代码块，尝试匹配 SELECT 语句
    select_match = re.search(r'SELECT.*?;', text, re.DOTALL)
    if select_match:
        return select_match.group(0).strip()
    
    # 如果都没有找到，返回原始文本
    return text.strip()

# 9. 执行 SQL 并返回结果
def execute_sql(sql: str):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            cols = result.keys()
            rows = result.fetchall()
            return True, cols, rows
    except Exception as e:
        return False, None, str(e)

# 10. 生成 SQL 函数
def generate_sql(prompt: str, error_msg: str = None) -> str:
    if error_msg:
        prompt += f"\n之前的SQL执行失败，错误信息：{error_msg}\n请修正SQL语句："
    
    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_sql = response.choices[0].message.content.strip()
    sql = extract_sql(raw_sql)
    logging.info(f"[生成] 原始输出: {raw_sql}")
    logging.info(f"[生成] 提取的SQL: {sql}")
    return sql

# 11. 核心流程：自然语言 -> SQL -> 执行 -> 返回
def text2sql(question: str, max_retries: int = 3):
    # 11.1 用户提问嵌入
    q_emb = embedding_fn([question])[0]
    logging.info(f"[检索] 问题嵌入完成")

    # 11.2 RAG 检索：DDL
    ddl_hits = retrieve("ddl_knowledge", q_emb.tolist(), top_k=3, fields=["ddl_text"])
    # logging.info(f"[检索] DDL检索结果: {ddl_hits}")
    try:
        ddl_context = "\n".join(hit.get("ddl_text", "") for hit in ddl_hits)
    except Exception as e:
        logging.error(f"[检索] DDL处理错误: {e}")
        ddl_context = ""

    # 11.3 RAG 检索：示例对
    q2sql_hits = retrieve("q2sql_knowledge", q_emb.tolist(), top_k=3, fields=["question", "sql_text"])
    # logging.info(f"[检索] Q2SQL检索结果: {q2sql_hits}")
    try:
        example_context = "\n".join(
            f"NL: \"{hit.get('question', '')}\"\nSQL: \"{hit.get('sql_text', '')}\"" 
            for hit in q2sql_hits
        )
    except Exception as e:
        logging.error(f"[检索] Q2SQL处理错误: {e}")
        example_context = ""

    # 11.4 RAG 检索：字段描述
    desc_hits = retrieve("dbdesc_knowledge", q_emb.tolist(), top_k=5, fields=["table_name", "column_name", "description"])
    # logging.info(f"[检索] 字段描述检索结果: {desc_hits}")
    try:
        desc_context = "\n".join(
            f"{hit.get('table_name', '')}.{hit.get('column_name', '')}: {hit.get('description', '')}"
            for hit in desc_hits
        )
    except Exception as e:
        logging.error(f"[检索] 字段描述处理错误: {e}")
        desc_context = ""

    # 11.5 组装基础 Prompt
    base_prompt = (
        f"### Schema Definitions:\n{ddl_context}\n"
        f"### Field Descriptions:\n{desc_context}\n"
        f"### Examples:\n{example_context}\n"
        f"### Query:\n\"{question}\"\n"
        "请只返回SQL语句，不要包含任何解释或说明。"
    )

    # 11.6 生成并执行 SQL，最多重试 max_retries 次
    error_msg = None
    for attempt in range(max_retries):
        logging.info(f"[执行] 第 {attempt + 1} 次尝试")
        
        # 生成 SQL
        sql = generate_sql(base_prompt, error_msg)
        
        # 执行 SQL
        success, cols, result = execute_sql(sql)
        
        if success:
            print("\n查询结果：")
            print("列名：", cols)
            for r in result:
                print(r)
            return
        
        error_msg = result
        logging.error(f"[执行] 第 {attempt + 1} 次执行失败: {error_msg}")
    
    print(f"执行失败，已达到最大重试次数 {max_retries}。")
    print("最后错误信息：", error_msg)

# 12. 程序入口
if __name__ == "__main__":
    user_q = input("请输入您的自然语言查询： ")
    text2sql(user_q)