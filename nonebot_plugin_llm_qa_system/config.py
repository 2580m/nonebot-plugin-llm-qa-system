"""nonebot_plugin_llm_qa_system - 配置"""

from pydantic import BaseModel, Extra


class Config(BaseModel, extra=Extra.ignore):
    """插件配置项，在 .env 文件中设置"""

    # Ollama 服务地址
    llm_qa_ollama_host: str = "http://localhost:11434"

    # 对话模型名称
    llm_qa_chat_model: str = "qwen3:1.7b"

    # 嵌入模型名称
    llm_qa_embed_model: str = "nomic-embed-text"

    # RAG 检索返回的最大相关文档数
    llm_qa_top_k: int = 3

    # 余弦相似度最低阈值，低于该值的条目不返回也不展示
    llm_qa_min_score: float = 0.3

    # 系统提示词
    llm_qa_system_prompt: str = (
        "你是一个智能问答助手。请根据提供的参考信息，"
        "用中文回答用户的问题。如果参考信息不足以回答问题，"
        "请如实告知，不要编造答案。"
    )
