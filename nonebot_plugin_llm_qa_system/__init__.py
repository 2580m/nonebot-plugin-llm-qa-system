"""nonebot_plugin_llm_qa_system - 基于 RAG 的智能问答系统

基于本地 Ollama 大模型 + 语义检索的知识问答机器人。

命令:
  问答 <问题>          — 基于知识库回答用户问题
  添加知识 <标题> <内容>  — 向知识库添加条目
  删除知识 <id>         — 删除指定知识条目
  列出知识              — 列出知识库所有条目
  清空知识              — 清空知识库（需确认）
  搜索知识 <关键词>      — 搜索知识库
  缓存状态              — 查看缓存统计
  清空缓存              — 清空所有缓存（需确认）
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from datetime import datetime

from nonebot import on_command, logger, require
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, permission as perm
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata

require("nonebot_plugin_orm")

from nonebot_plugin_orm import get_session as get_orm_session
from sqlalchemy import delete, select, func

from .config import Config
from .models import KnowledgeEntry, QACache, EmbeddingCache, AnswerCache
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
        "清空知识 — 清空全部（需确认）\n"
        "缓存状态 — 查看缓存统计\n"
        "清空缓存 — 清空所有缓存（需确认）"
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

    # 查询缓存（归一化后精确匹配）
    normalized_query = _normalize_query(query)
    async with get_orm_session() as session:
        stmt = select(QACache).where(QACache.query == normalized_query)
        cached = (await session.execute(stmt)).scalar_one_or_none()
        if cached:
            sources = json.loads(cached.sources_json)
            reply = (
                f"💡 回答：\n{cached.answer}\n\n"
                f"📎 参考来源：\n" + "\n".join(sources)
            )
            await qa_cmd.finish(reply)

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

    # 检索（遇到空嵌入会自动重新生成并更新 entry_dicts）
    await qa_cmd.send("📚 正在检索相关知识...")
    try:
        relevant = await engine.retrieve(query, entry_dicts)
    except Exception as e:
        logger.error(f"llm_qa: 检索失败: {e}")
        await qa_cmd.finish(f"❌ 检索失败：{e}")
        return

    # 将自动重新生成的空嵌入持久化到数据库
    stale_ids = [
        d["id"] for d, e in zip(entry_dicts, entries)
        if d["embedding"] != e.embedding and d["embedding"] not in ("[]", "", None)
    ]
    if stale_ids:
        async with get_orm_session() as session:
            for d in entry_dicts:
                if d["id"] in stale_ids:
                    stmt = select(KnowledgeEntry).where(KnowledgeEntry.id == d["id"])
                    db_entry = (await session.execute(stmt)).scalar_one_or_none()
                    if db_entry:
                        db_entry.embedding = d["embedding"]
            await session.commit()

    if not relevant:
        await qa_cmd.finish("未找到相关问题，请尝试换一种问法。")

    # 构建上下文文本，计算回答缓存键
    context_parts: list[str] = []
    for c in relevant:
        context_parts.append(f"[{c.get('title', '未知')}]\n{c.get('content', '')}")
    context_text = "\n\n".join(context_parts)

    cache_key = hashlib.md5(
        f"知识库:\n{context_text}\n问题:\n{normalized_query}".encode()
    ).hexdigest()

    async with get_orm_session() as session:
        cached_ans = (
            await session.execute(
                select(AnswerCache).where(AnswerCache.cache_key == cache_key)
            )
        ).scalar_one_or_none()
        if cached_ans:
            sources = json.loads(cached_ans.sources_json)
            reply = (
                f"💡 回答：\n{cached_ans.answer}\n\n"
                f"📎 参考来源：\n" + "\n".join(sources)
            )
            await qa_cmd.finish(reply)

    # 生成回答
    await qa_cmd.send("🤖 正在生成回答...")
    answer = await engine.ask(query, relevant)

    # 构建回复
    source_lines = [f"[{i+1}] {c.get('title', '未知')}" for i, c in enumerate(relevant)]
    reply = (
        f"💡 回答：\n{answer}\n\n"
        f"📎 参考来源：\n" + "\n".join(source_lines)
    )

    # 写入各种缓存
    async with get_orm_session() as session:
        # QACache（归一化 query 精确匹配）
        existing = (await session.execute(
            select(QACache).where(QACache.query == normalized_query)
        )).scalar_one_or_none()
        if existing:
            existing.answer = answer
            existing.sources_json = json.dumps(source_lines)
            existing.created_at = datetime.now()
        else:
            session.add(QACache(query=normalized_query, answer=answer, sources_json=json.dumps(source_lines)))

        # AnswerCache（Prompt 哈希匹配）
        existing_ans = (
            await session.execute(
                select(AnswerCache).where(AnswerCache.cache_key == cache_key)
            )
        ).scalar_one_or_none()
        if existing_ans:
            existing_ans.question = query
            existing_ans.answer = answer
            existing_ans.sources_json = json.dumps(source_lines)
            existing_ans.created_at = datetime.now()
        else:
            session.add(AnswerCache(
                cache_key=cache_key,
                question=query,
                answer=answer,
                sources_json=json.dumps(source_lines),
            ))

        await session.commit()

    await qa_cmd.finish(reply)


async def _clear_all_cache() -> None:
    """知识变更后清空问答/回答缓存（嵌入缓存保留，其不依赖知识内容）。"""
    async with get_orm_session() as session:
        await session.execute(delete(QACache))
        await session.execute(delete(AnswerCache))
        await session.commit()
    # 同时清空引擎内进程级缓存
    engine = await _get_engine()
    engine._embed_cache.clear()


def _normalize_query(text: str) -> str:
    """归一化查询文本，使语义相同但表述不同的问题命中同一缓存。"""
    text = re.sub(r"\s+", " ", text).strip()
    for prefix in ("请", "请问", "帮我", "帮我一下"):
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
            break
    text = text.rstrip("？?。.！!，,")
    return text


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

    # 生成嵌入（优先查嵌入缓存）
    embed_text = f"{title}\n{content}"
    async with get_orm_session() as session:
        cached_emb = (
            await session.execute(
                select(EmbeddingCache).where(EmbeddingCache.text == embed_text)
            )
        ).scalar_one_or_none()

    if cached_emb:
        embedding = json.loads(cached_emb.embedding)
    else:
        engine = await _get_engine()
        try:
            embedding = await engine.embed(embed_text)
        except Exception as e:
            logger.error(f"llm_qa: 生成嵌入失败: {e}")
            await add_cmd.finish(f"❌ 生成嵌入向量失败，无法添加知识：{e}")
            return

        if not embedding:
            await add_cmd.finish("❌ 嵌入向量返回为空，请检查 Ollama 嵌入模型是否可用。")
            return

        # 持久化嵌入缓存
        async with get_orm_session() as session:
            session.add(EmbeddingCache(text=embed_text, embedding=json.dumps(embedding)))
            await session.commit()

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

    # 清空缓存，下次问答重新生成
    await _clear_all_cache()

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
    """列出知识库所有条目（分页，每页最多 10 条或 1500 字符）。"""
    async with get_orm_session() as session:
        stmt = select(KnowledgeEntry).order_by(KnowledgeEntry.id)
        result = await session.execute(stmt)
        entries = result.scalars().all()

    if not entries:
        await list_cmd.finish("📭 知识库为空")

    # 构建条目行
    entry_lines: list[str] = []
    for i, e in enumerate(entries, 1):
        preview = e.content[:80].replace("\n", " ")
        entry_lines.append(
            f"  #{e.id} [{i}] {e.title} — {preview}{'...' if len(e.content) > 80 else ''}"
        )

    total = len(entries)

    # 分页：每页最多 10 条或 1500 字符（取先达到者）
    pages: list[list[str]] = []
    page: list[str] = []
    page_len = 0
    for line in entry_lines:
        add_cost = len(line) + (1 if page else 0)  # 换行符
        if (
            len(page) >= 10
            or (page and page_len + add_cost > 1500)
        ):
            pages.append(page)
            page = []
            page_len = 0
        page.append(line)
        page_len += len(line) + (1 if len(page) > 1 else 0)
    if page:
        pages.append(page)

    # 发送各页
    header = "📚 知识库列表："
    for idx, page_lines in enumerate(pages):
        footer = f"\n— 第 {idx + 1}/{len(pages)} 页，共 {total} 条 —"
        msg = "\n".join([header] + page_lines + [footer])
        if idx == len(pages) - 1:
            await list_cmd.finish(msg)
        else:
            await list_cmd.send(msg)


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

    # 持久化自动补全的空嵌入
    stale_ids = [
        d["id"] for d, e in zip(entry_dicts, entries)
        if d["embedding"] != e.embedding and d["embedding"] not in ("[]", "", None)
    ]
    if stale_ids:
        async with get_orm_session() as session:
            for d in entry_dicts:
                if d["id"] in stale_ids:
                    stmt = select(KnowledgeEntry).where(KnowledgeEntry.id == d["id"])
                    db_entry = (await session.execute(stmt)).scalar_one_or_none()
                    if db_entry:
                        db_entry.embedding = d["embedding"]
            await session.commit()

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

    await _clear_all_cache()

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

    await _clear_all_cache()

    await clear_cmd.finish(f"🗑️ 已清空知识库，共删除 {count} 条")


# ==================== 缓存管理 ====================

cache_status_cmd = on_command("缓存状态", permission=SUPERUSER, priority=10, block=True)


@cache_status_cmd.handle()
async def handle_cache_status(
    bot: Bot,
    event: GroupMessageEvent,
) -> None:
    """查看缓存状态。"""
    async with get_orm_session() as session:
        qa_count = (await session.execute(select(func.count()).select_from(QACache))).scalar()
        emb_count = (await session.execute(select(func.count()).select_from(EmbeddingCache))).scalar()
        ans_count = (await session.execute(select(func.count()).select_from(AnswerCache))).scalar()
        mem_count = len((await _get_engine())._embed_cache)

    await cache_status_cmd.finish(
        "📊 缓存状态：\n"
        f"  问答缓存（精确匹配）：{qa_count} 条\n"
        f"  回答缓存（哈希匹配）：{ans_count} 条\n"
        f"  嵌入缓存（持久化）：  {emb_count} 条\n"
        f"  嵌入缓存（进程级）：  {mem_count} 条"
    )


clear_cache_cmd = on_command("清空缓存", permission=SUPERUSER, priority=10, block=True)


@clear_cache_cmd.handle()
async def handle_clear_cache(
    bot: Bot,
    event: GroupMessageEvent,
    args: Any = CommandArg(),
) -> None:
    """清空所有缓存。"""
    confirm = args.extract_plain_text().strip()
    if confirm != "确认":
        await clear_cache_cmd.finish(
            "⚠️ 确认要清空所有缓存吗？\n"
            "此操作将清空问答缓存和回答缓存。\n"
            "请发送：清空缓存 确认"
        )

    await _clear_all_cache()
    await clear_cache_cmd.finish("🗑️ 已清空所有缓存")

from nonebot import get_driver

driver = get_driver()


@driver.on_shutdown
async def _():
    if _engine is not None:
        await _engine.close()
