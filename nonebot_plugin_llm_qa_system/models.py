"""nonebot_plugin_llm_qa_system - ORM 数据模型"""

from nonebot import require

require("nonebot_plugin_orm")

from nonebot_plugin_orm import Model
from sqlalchemy import TEXT, Integer, String
from sqlalchemy.orm import Mapped, mapped_column


class KnowledgeEntry(Model):
    """知识条目表"""

    __tablename__ = "llm_qa_knowledge"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), comment="标题/关键词")
    content: Mapped[str] = mapped_column(TEXT, comment="知识内容")
    embedding: Mapped[str] = mapped_column(TEXT, comment="嵌入向量（JSON数组）", default="[]")
