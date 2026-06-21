"""nonebot_plugin_llm_qa_system - 基于 RAG 的智能问答系统

基于本地 Ollama 大模型 + 语义检索的知识问答机器人。

命令:
  问答 <问题>          — 基于知识库回答用户问题
  添加知识 <标题> <内容>  — 向知识库添加条目
  删除知识 <id>         — 删除指定知识条目
  列出知识              — 列出知识库所有条目
  清空知识              — 清空知识库（需确认）
  搜索知识 <关键词>      — 搜索知识库
"""

from __future__ import annotations

import json
from typing import Any

from nonebot import on_command, logger, require
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, permission as perm
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata

require("nonebot_plugin_orm")

from nonebot_plugin_orm import get_session as get_orm_session
from sqlalchemy import delete, select

from .config import Config
from .models import KnowledgeEntry
from .rag_engine import RAGEngine

__plugin_meta__ = PluginMetadata(
    name="nonebot-plugin-llm-qa-system",
    description="基于 Ollama + RAG 的智能问答系统",
    usage=(
        "问答 <问题> — 基于知识库回答\n"
        "添加知识 <标题> <内容> — 添加知识条目\n"
        "删除知识 <id> — 删除指定条目\n"
        "列出知识 — 列出所有条目\n"
        "搜索知识 <关键词> — 语义搜索\n"
        "清空知识 — 清空全部（需确认）"
    ),
    type="application",
    config=Config,
)

# ==================== 配置加载 ====================

try:
    from nonebot import get_plugin_config
    plugin_config = get_plugin_config(Config)
except ImportError:
    from nonebot import get_driver
    plugin_config = Config.parse_obj(get_driver().config)

# ==================== 全局引擎 ====================

_engine: RAGEngine | None = None


async def _get_engine() -> RAGEngine:
    """获取或初始化 RAG 引擎。"""
    global _engine
    if _engine is None:
        _engine = RAGEngine(plugin_config)
    return _engine


# ==================== 问答命令 ====================

qa_cmd = on_command("问答", permission=perm.GROUP, priority=10, block=True)


@qa_cmd.handle()
async def handle_qa(
    bot: Bot,
    event: GroupMessageEvent,
    args: Any = CommandArg(),
) -> None:
    """基于知识库回答用户问题。"""
    query = args.extract_plain_text().strip()
    if not query:
        await qa_cmd.finish("用法：问答 <你的问题>")

    await qa_cmd.send(f"🔍 正在思考：{query}")

    # 加载知识库
    async with get_orm_session() as session:
        stmt = select(KnowledgeEntry)
        result = await session.execute(stmt)
        entries = result.scalars().all()

    if not entries:
        await qa_cmd.finish("知识库为空，请先添加知识。\n用法：添加知识 <标题> <内容>")

    # 转为 dict 供检索
    entry_dicts = [
        {
            "id": e.id,
            "title": e.title,
            "content": e.content,
            "embedding": e.embedding,
        }
        for e in entries
    ]

    engine = await _get_engine()

    # 检索
    await qa_cmd.send("📚 正在检索相关知识...")
    try:
        relevant = await engine.retrieve(query, entry_dicts)
    except Exception as e:
        logger.error(f"llm_qa: 检索失败: {e}")
        await qa_cmd.finish(f"❌ 检索失败：{e}")
        return

    if not relevant:
        await qa_cmd.finish("未找到相关问题，请尝试换一种问法。")

    # 生成回答
    await qa_cmd.send("🤖 正在生成回答...")
    answer = await engine.ask(query, relevant)

    # 构建回复
    sources = [f"[{i+1}] {c.get('title', '未知')}" for i, c in enumerate(relevant)]
    reply = (
        f"💡 回答：\n{answer}\n\n"
        f"📎 参考来源：\n" + "\n".join(sources)
    )
    await qa_cmd.finish(reply)


# ==================== 知识管理命令 ====================

add_cmd = on_command("添加知识", permission=perm.GROUP, priority=10, block=True)


@add_cmd.handle()
async def handle_add_knowledge(
    bot: Bot,
    event: GroupMessageEvent,
    args: Any = CommandArg(),
) -> None:
    """添加知识条目。"""
    text = args.extract_plain_text().strip()
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        await add_cmd.finish("用法：添加知识 <标题> <内容>")

    title = parts[0]
    content = parts[1]

    # 生成嵌入
    engine = await _get_engine()
    try:
        embedding = await engine.embed(f"{title}\n{content}")
    except Exception as e:
        logger.error(f"llm_qa: 生成嵌入失败: {e}")
        await add_cmd.finish(f"❌ 生成嵌入向量失败，无法添加知识：{e}")
        return

    if not embedding:
        await add_cmd.finish("❌ 嵌入向量返回为空，请检查 Ollama 嵌入模型是否可用。")
        return

    # 入库
    async with get_orm_session() as session:
        entry = KnowledgeEntry(
            title=title,
            content=content,
            embedding=json.dumps(embedding),
        )
        session.add(entry)
        await session.flush()  # 先 flush 让数据库生成自增 ID
        entry_id = entry.id    # flush 后 id 已填充到实例中，此时访问不会触发 lazy load
        await session.commit()

    await add_cmd.finish(
        f"✅ 已添加知识 #{entry_id}\n"
        f"标题：{title}\n"
        f"内容：{content[:100]}{'...' if len(content) > 100 else ''}"
    )


# ==================== 列出知识 ====================

list_cmd = on_command("列出知识", permission=perm.GROUP, priority=10, block=True)


@list_cmd.handle()
async def handle_list_knowledge(
    bot: Bot,
    event: GroupMessageEvent,
) -> None:
    """列出知识库所有条目。"""
    async with get_orm_session() as session:
        stmt = select(KnowledgeEntry).order_by(KnowledgeEntry.id)
        result = await session.execute(stmt)
        entries = result.scalars().all()

    if not entries:
        await list_cmd.finish("📭 知识库为空")

    lines = ["📚 知识库列表："]
    for i, e in enumerate(entries, 1):
        preview = e.content[:80].replace("\n", " ")
        lines.append(f"  #{e.id} [{i}] {e.title} — {preview}{'...' if len(e.content) > 80 else ''}")
    lines.append(f"\n共 {len(entries)} 条")

    # 分批发送避免消息过长
    msg = "\n".join(lines)
    if len(msg) > 1500:
        chunks = []
        current = []
        for line in lines:
            if current and len("\n".join(current + [line])) > 1000:
                chunks.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            chunks.append("\n".join(current))
        # 除最后一条外都用 send，最后一条用 finish
        for chunk in chunks[:-1]:
            await list_cmd.send(chunk)
        await list_cmd.finish(chunks[-1])
    else:
        await list_cmd.finish(msg)


# ==================== 搜索知识 ====================

search_cmd = on_command("搜索知识", permission=perm.GROUP, priority=10, block=True)


@search_cmd.handle()
async def handle_search_knowledge(
    bot: Bot,
    event: GroupMessageEvent,
    args: Any = CommandArg(),
) -> None:
    """语义搜索知识库。"""
    query = args.extract_plain_text().strip()
    if not query:
        await search_cmd.finish("用法：搜索知识 <关键词>")

    async with get_orm_session() as session:
        stmt = select(KnowledgeEntry)
        result = await session.execute(stmt)
        entries = result.scalars().all()

    if not entries:
        await search_cmd.finish("📭 知识库为空")

    entry_dicts = [
        {"id": e.id, "title": e.title, "content": e.content, "embedding": e.embedding}
        for e in entries
    ]

    engine = await _get_engine()
    try:
        relevant = await engine.retrieve(query, entry_dicts, top_k=5)
    except Exception as e:
        logger.error(f"llm_qa: 搜索失败: {e}")
        await search_cmd.finish(f"❌ 搜索失败：{e}")
        return

    if not relevant:
        await search_cmd.finish(f"未找到与「{query}」相关的内容")

    lines = [f"🔍 搜索「{query}」结果："]
    for i, c in enumerate(relevant, 1):
        content_preview = c["content"][:100].replace("\n", " ")
        lines.append(f"  #{c['id']} [{i}] {c['title']}")
        lines.append(f"     {content_preview}{'...' if len(c['content']) > 100 else ''}")

    await search_cmd.finish("\n".join(lines))


# ==================== 删除知识 ====================

del_cmd = on_command("删除知识", permission=SUPERUSER, priority=10, block=True)


@del_cmd.handle()
async def handle_delete_knowledge(
    bot: Bot,
    event: GroupMessageEvent,
    args: Any = CommandArg(),
) -> None:
    """删除指定知识条目。"""
    text = args.extract_plain_text().strip()
    if not text.isdigit():
        await del_cmd.finish("用法：删除知识 <ID>")

    entry_id = int(text)

    async with get_orm_session() as session:
        stmt = select(KnowledgeEntry).where(KnowledgeEntry.id == entry_id)
        result = await session.execute(stmt)
        entry = result.scalar_one_or_none()
        if entry is None:
            await del_cmd.finish(f"❌ 未找到 ID 为 {entry_id} 的知识条目")

        title = entry.title
        await session.delete(entry)
        await session.commit()

    await del_cmd.finish(f"🗑️ 已删除 #{entry_id} {title}")


# ==================== 清空知识 ====================

clear_cmd = on_command("清空知识", permission=SUPERUSER, priority=10, block=True)


@clear_cmd.handle()
async def handle_clear_knowledge(
    bot: Bot,
    event: GroupMessageEvent,
    args: Any = CommandArg(),
) -> None:
    """清空知识库。"""
    confirm = args.extract_plain_text().strip()
    if confirm != "确认":
        await clear_cmd.finish(
            "⚠️ 确认要清空所有知识条目吗？\n"
            "此操作不可撤销。\n"
            "请发送：清空知识 确认"
        )

    async with get_orm_session() as session:
        stmt = delete(KnowledgeEntry)
        result = await session.execute(stmt)
        await session.commit()
        count = result.rowcount

    await clear_cmd.finish(f"🗑️ 已清空知识库，共删除 {count} 条")


# ==================== 启动/关闭事件 ====================

from nonebot import get_driver

driver = get_driver()


@driver.on_shutdown
async def _():
    if _engine is not None:
        await _engine.close()
