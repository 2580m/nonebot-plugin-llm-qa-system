"""nonebot_plugin_llm_qa_system - ORM 数据模型"""

from datetime import datetime

from nonebot import require

require("nonebot_plugin_orm")

from nonebot_plugin_orm import Model
from sqlalchemy import TEXT, DateTime, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column


class KnowledgeEntry(Model):
    """知识条目表"""

    __tablename__ = "llm_qa_knowledge"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), comment="标题/关键词")
    content: Mapped[str] = mapped_column(TEXT, comment="知识内容")
    embedding: Mapped[str] = mapped_column(TEXT, comment="嵌入向量（JSON数组）", default="[]")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, comment="最后更新/创建时间")


class EmbeddingCache(Model):
    """嵌入向量缓存（文本→向量），按模型名隔离。"""

    __tablename__ = "llm_qa_embedding_cache"
    __table_args__ = (
        UniqueConstraint("text", "model_name", name="uq_embedding_text_model"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(TEXT, comment="文本原文")
    model_name: Mapped[str] = mapped_column(String(64), comment="嵌入模型名")
    embedding: Mapped[str] = mapped_column(TEXT, comment="嵌入向量（JSON数组）")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, comment="缓存创建时间"
    )


class AnswerCache(Model):
    """回答缓存（基于 Prompt Hash）"""

    __tablename__ = "llm_qa_answer_cache"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    cache_key: Mapped[str] = mapped_column(String(64), unique=True, comment="SHA256 缓存键")
    question: Mapped[str] = mapped_column(TEXT, comment="用户问题")
    answer: Mapped[str] = mapped_column(TEXT, comment="LLM 生成的回答")
    sources_json: Mapped[str] = mapped_column(TEXT, comment="参考来源标题列表（JSON数组）")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, comment="缓存创建时间"
    )


class KnowledgeVersion(Model):
    """知识库版本号，知识变更时递增。"""

    __tablename__ = "llm_qa_knowledge_version"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # 固定为 1
    version: Mapped[int] = mapped_column(Integer, default=0, comment="知识库版本号，变更时递增")


class SemanticCache(Model):
    """语义缓存（基于向量相似度匹配）"""

    __tablename__ = "llm_qa_semantic_cache"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(TEXT, comment="归一化后的问题")
    query_embedding: Mapped[str] = mapped_column(TEXT, comment="问题嵌入向量（JSON数组）")
    answer: Mapped[str] = mapped_column(TEXT, comment="LLM 生成的回答")
    knowledge_version: Mapped[int] = mapped_column(Integer, comment="写入时的知识库版本号")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, comment="缓存创建时间"
    )
