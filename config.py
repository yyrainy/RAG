
# 全局变量
vectordb = None
db_status_components = []

# 常量配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
TOP_N = 3
EMBEDDING_MODEL = "./models/all-MiniLM-L6-v2"
RERANK_MODEL = "./models/ms-marco-MiniLM-L-6-v2"
PERSIST_DIR = "./faiss_db_cpu"

# 模型简单参数
temperature = 0.7
max_token = 1024
# timeout = 60
# max_retry = 1
api_key="xxxxxxx"
# LLM配置
LLM_CANDIDATES = [
    {"model_id": "ZhipuAI/GLM-4.7-Flash", "name": "GLM-4.7-Flash"},
    {"model_id": "Qwen/Qwen-7B-Chat", "name": "通义千问7B"},
    {"model_id": "ZhipuAI/GLM-3-Turbo", "name": "GLM-3-Turbo"},
    {"model_id":"MiniMax/MiniMax-M2.5","name":"MinMax"},
    {"model_id": "moonshotai/Kimi-K2.5", "name": "moonshotai"}
]
#用来更新chat_history防止报错
from langchain_community.chat_message_histories import ChatMessageHistory
DEFAULT_SESSION_ID = "session_001"
SESSION_STORAGE = {
    DEFAULT_SESSION_ID: {"name": "新会话", "chat_history": ChatMessageHistory(),"visible":False}
}