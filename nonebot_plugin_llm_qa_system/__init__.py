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
from collections import OrderedDict
from typing import Any

from datetime import datetime, timedelta

import jieba_next

from nonebot import on_command, logger, require
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, permission as perm
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata

require("nonebot_plugin_orm")

from nonebot_plugin_orm import get_session as get_orm_session
from sqlalchemy import delete, select, func

from .config import Config
from .models import KnowledgeEntry, EmbeddingCache, AnswerCache, KnowledgeVersion, SemanticCache
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

# ==================== 内存 LRU 缓存 ====================


class LRUCache:
    """内存 LRU 缓存，达到上限时淘汰最久未使用的条目。"""

    def __init__(self, maxsize: int = 100) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[str, tuple[str, str]] = OrderedDict()

    def get(self, key: str) -> tuple[str, str] | None:
        """读取缓存，命中则将该条目移至末尾。"""
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, answer: str, sources_json: str) -> None:
        """写入缓存，超出上限时淘汰最久未使用的条目。"""
        self._cache[key] = (answer, sources_json)
        self._cache.move_to_end(key)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """清空缓存。"""
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


qa_lru_cache = LRUCache(maxsize=100)


class CacheStats:
    """进程级缓存命中统计。"""

    def __init__(self) -> None:
        self.embed_hit = 0
        self.embed_miss = 0
        self.answer_hit = 0
        self.answer_miss = 0
        self.semantic_hit = 0
        self.semantic_miss = 0

    @property
    def total_hits(self) -> int:
        return self.embed_hit + self.answer_hit + self.semantic_hit

    @property
    def total_misses(self) -> int:
        return self.embed_miss + self.answer_miss + self.semantic_miss

    @property
    def total(self) -> int:
        return self.total_hits + self.total_misses

    @property
    def hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.total_hits / self.total * 100


cache_stats = CacheStats()


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
    # 归一化查询文本；若全部被过滤则回退到原始输入（避免空 key 污染缓存）
    normalized_query = _normalize_query(query) or query
    cached = qa_lru_cache.get(normalized_query)
    if cached:
        answer, sources_json = cached
        sources = json.loads(sources_json)
        reply = (
            f"💡 回答：\n{answer}\n\n"
            f"📎 参考来源：\n" + "\n".join(sources)
        )
        await qa_cmd.finish(reply)

    await qa_cmd.send(f"🔍 正在思考：{query}")

    # 获取知识库版本和查询向量，用于语义缓存
    async with get_orm_session() as session:
        knowledge_version = await _get_knowledge_version(session)
        query_embedding = await _get_or_compute_embedding(normalized_query, session)

    # 语义缓存检查（同一版本号，最近 200 条）
    async with get_orm_session() as session:
        candidates = (
            await session.execute(
                select(SemanticCache)
                .where(SemanticCache.knowledge_version == knowledge_version)
                .order_by(SemanticCache.created_at.desc())
                .limit(plugin_config.semantic_cache_max_candidates)
            )
        ).scalars().all()

    best_sim = 0.0
    best_answer = None
    best_cached_query = None
    for c in candidates:
        emb = json.loads(c.query_embedding)
        sim = RAGEngine.cosine_similarity(query_embedding, emb)
        if sim > best_sim:
            best_sim = sim
            best_answer = c.answer
            best_cached_query = c.query

    if best_sim >= plugin_config.semantic_cache_threshold:
        cache_stats.semantic_hit += 1
        logger.info(
            f"llm_qa: SemanticCache HIT | "
            f"query={normalized_query!r} | "
            f"cached_query={best_cached_query!r} | "
            f"score={best_sim:.4f} | "
            f"version={knowledge_version}"
        )
        await qa_cmd.finish(f"💡 回答（语义缓存）：\n{best_answer}")
    cache_stats.semantic_miss += 1

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

    # 构造知识指纹（基于内容的哈希，不依赖数据库序号）
    content_hashes = sorted(
        hashlib.sha256(f"{c['title']}\n{c['content']}".encode()).hexdigest()
        for c in relevant
    )
    knowledge_fingerprint = "|".join(content_hashes)

    cache_key = hashlib.sha256(
        f"{knowledge_version}|"
        f"{plugin_config.llm_qa_system_prompt}|"
        f"{plugin_config.llm_qa_chat_model}|"
        f"{knowledge_fingerprint}|"
        f"{normalized_query}".encode()
    ).hexdigest()

    async with get_orm_session() as session:
        cached_ans = (
            await session.execute(
                select(AnswerCache).where(AnswerCache.cache_key == cache_key)
            )
        ).scalar_one_or_none()
        if cached_ans:
            cache_stats.answer_hit += 1
            # AnswerCache 命中时同步写入语义缓存作为训练样本
            await _upsert_semantic_cache(
                session,
                query=normalized_query,
                query_embedding=query_embedding,
                answer=cached_ans.answer,
                knowledge_version=knowledge_version,
            )
            await session.commit()
            sources = json.loads(cached_ans.sources_json)
            reply = (
                f"💡 回答：\n{cached_ans.answer}\n\n"
                f"📎 参考来源：\n" + "\n".join(sources)
            )
            await qa_cmd.finish(reply)

    # 生成回答
    cache_stats.answer_miss += 1
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
        # QACache（归一化 query 精确匹配，内存 LRU）
        qa_lru_cache.put(normalized_query, answer, json.dumps(source_lines))

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

    # 写入语义缓存（幂等 upsert）
    async with get_orm_session() as session:
        await _upsert_semantic_cache(
            session,
            query=normalized_query,
            query_embedding=query_embedding,
            answer=answer,
            knowledge_version=knowledge_version,
        )
        await session.commit()

    # 定期清理过期语义缓存
    await _cleanup_semantic_cache()

    await qa_cmd.finish(reply)


async def _clear_all_cache() -> None:
    """知识变更后清空内存问答缓存和进程级嵌入缓存。
    AnswerCache 基于内容哈希，知识内容不变时自动命中，不清除。"""
    qa_lru_cache.clear()
    engine = await _get_engine()
    engine._embed_cache.clear()


async def _get_knowledge_version(session) -> int:
    """读取当前知识库版本号，首次调用时初始化为 0。"""
    stmt = select(KnowledgeVersion).where(KnowledgeVersion.id == 1)
    kv = (await session.execute(stmt)).scalar_one_or_none()
    if kv is None:
        kv = KnowledgeVersion(id=1, version=0)
        session.add(kv)
        await session.flush()
    return kv.version


async def _increment_knowledge_version(session) -> int:
    """知识变更时递增版本号。"""
    stmt = select(KnowledgeVersion).where(KnowledgeVersion.id == 1)
    kv = (await session.execute(stmt)).scalar_one_or_none()
    if kv is None:
        kv = KnowledgeVersion(id=1, version=0)
        session.add(kv)
    kv.version += 1
    await session.flush()
    return kv.version


async def _get_or_compute_embedding(text: str, session) -> list[float]:
    """从 EmbeddingCache 读取嵌入向量，未命中则调用 Ollama 并持久化。"""
    stmt = select(EmbeddingCache).where(
        EmbeddingCache.text == text,
        EmbeddingCache.model_name == plugin_config.llm_qa_embed_model,
    )
    cached = (await session.execute(stmt)).scalar_one_or_none()
    if cached:
        cache_stats.embed_hit += 1
        return json.loads(cached.embedding)

    cache_stats.embed_miss += 1
    engine = await _get_engine()
    embedding = await engine.embed(text)
    session.add(EmbeddingCache(
        text=text,
        model_name=plugin_config.llm_qa_embed_model,
        embedding=json.dumps(embedding),
    ))
    await session.flush()
    return embedding


async def _cleanup_semantic_cache() -> None:
    """清理过期的语义缓存：删除 30 天前或版本差超过 5 的条目。"""
    async with get_orm_session() as session:
        current_version = await _get_knowledge_version(session)
        cutoff = datetime.now() - timedelta(days=30)
        await session.execute(
            delete(SemanticCache).where(SemanticCache.created_at < cutoff)
        )
        await session.execute(
            delete(SemanticCache).where(
                SemanticCache.knowledge_version < current_version - 5
            )
        )
        await session.commit()


async def _upsert_semantic_cache(
    session,
    query: str,
    query_embedding: list[float],
    answer: str,
    knowledge_version: int,
) -> None:
    """写入语义缓存，同 query + version 已存在时更新（幂等）。"""
    stmt = select(SemanticCache).where(
        SemanticCache.query == query,
        SemanticCache.knowledge_version == knowledge_version,
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if existing:
        existing.query_embedding = json.dumps(query_embedding)
        existing.answer = answer
        existing.created_at = datetime.now()
    else:
        session.add(SemanticCache(
            query=query,
            query_embedding=json.dumps(query_embedding),
            answer=answer,
            knowledge_version=knowledge_version,
        ))


# 分词前替换的复合停用短语（按长度降序，避免短短语先替换破坏长短语匹配）
_COMPOUND_STOPWORDS = sorted([
    "我想问一下", "我想咨询一下", "我想了解一下",
    "帮我看一下", "帮我查一下", "帮我确认一下",
    "顺便问一下", "再问一下", "另外问一下",
    "请帮我", "能帮我", "可以帮我", "麻烦帮我",
    "一般来说", "正常来说",
    "的话呢", "就是说",
    "想请问一下", "麻烦你了", "辛苦你了",
    "打扰一下", "请问一下",
    "方便的话", "顺便问下",
    "各位好", "老师好",
    "有空吗", "在吗",
    "想问一下", "咨询一下", "了解一下",
    "帮我看", "帮我查", "帮我确认",
    "顺便问", "劳烦你",
    "我想", "想问",
    "打扰了", "辛苦了",
    "如下", "如上", "如题", "见上", "见下",
    "附件", "附上",
    "的话", "您好", "你好",
    "请问", "麻烦", "劳烦",
    "帮忙",
], key=len, reverse=True)


def _normalize_query(text: str) -> str:
    """归一化查询文本：分词 + 去停用词 + 重拼接，使不同表述的同类问题命中同一缓存。"""
    text = text.strip().rstrip("？?。.！!，,；;：:")
    if not text:
        return ""

    # 分词前先去掉复合停用短语（jieba 无法将这些整体识别为单词元）
    for phrase in _COMPOUND_STOPWORDS:
        text = text.replace(phrase, "")
    text = re.sub(r"\s+", " ", text).strip()

    tokens = jieba_next.lcut(text)

    stopwords = {
        "一", "一个", "一些", "一下", "一种", "上", "下面", "与", "且",
        "个", "中", "为", "之", "也", "了", "于", "些", "人", "他", "以",
        "们", "会", "但", "你", "来", "例如", "做", "像", "其", "再", "则",
        "刚", "到", "又", "及", "可", "可是", "让", "说", "请", "还",
        "这", "那", "都", "要", "见上", "见下", "认为",
        "给", "自己", "被", "把", "按照",
        "啊", "哎呀", "哎哟", "唉", "诶", "欸", "哦", "噢", "喔", "呵",
        "啦", "呀", "哟", "哇", "嗯", "恩", "额", "呢", "嘛", "吧", "呗",
        "哈",
        "在", "因", "因为", "所以", "而", "等", "从而", "从", "向",
        "关于", "其", "其中", "与否",
        "能", "能够", "可能",
        "已经", "已",
        "并", "并且", "以及", "还有", "另外", "同时",
        "对", "对于", "将", "就是", "其实", "然后",
        "相关", "内容", "情况", "事项", "部分", "方面", "东西",
        "谢谢", "感谢",
        "大概", "比较", "稍微", "有点",
        "我想", "想问", "咨询",
        "一下", "一个", "一些", "一种",
        "进行", "就是", "那个", "这个",
        "的话", "的话呢", "就是说",
        "一般来说", "正常来说",
    }

    filtered = [t for t in tokens if t not in stopwords and len(t) > 0]
    return " ".join(filtered) if filtered else text


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
                select(EmbeddingCache).where(
                    EmbeddingCache.text == embed_text,
                    EmbeddingCache.model_name == plugin_config.llm_qa_embed_model,
                )
            )
        ).scalar_one_or_none()

    if cached_emb:
        cache_stats.embed_hit += 1
        embedding = json.loads(cached_emb.embedding)
    else:
        cache_stats.embed_miss += 1
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
            session.add(EmbeddingCache(
                text=embed_text,
                model_name=plugin_config.llm_qa_embed_model,
                embedding=json.dumps(embedding),
            ))
            await session.commit()

    # 入库
    async with get_orm_session() as session:
        entry = KnowledgeEntry(
            title=title,
            content=content,
            embedding=json.dumps(embedding),
            updated_at=datetime.now(),
        )
        session.add(entry)
        await session.flush()  # 先 flush 让数据库生成自增 ID
        entry_id = entry.id    # flush 后 id 已填充到实例中，此时访问不会触发 lazy load
        await _increment_knowledge_version(session)
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
        await _increment_knowledge_version(session)
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
        await _increment_knowledge_version(session)
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
    qa_count = qa_lru_cache.size
    async with get_orm_session() as session:
        emb_count = (await session.execute(select(func.count()).select_from(EmbeddingCache))).scalar()
        ans_count = (await session.execute(select(func.count()).select_from(AnswerCache))).scalar()
        knowledge_version = await _get_knowledge_version(session)
        mem_count = len((await _get_engine())._embed_cache)

    await cache_status_cmd.finish(
        "📊 缓存状态：\n"
        f"  知识库版本：        {knowledge_version}\n"
        f"  问答缓存（LRU 精确匹配）：{qa_count} 条\n"
        f"  回答缓存（哈希匹配）：{ans_count} 条\n"
        f"  嵌入缓存（持久化）：  {emb_count} 条\n"
        f"  嵌入缓存（进程级）：  {mem_count} 条\n"
        f"\n"
        f"📈 缓存命中统计：\n"
        f"  Embedding 缓存: 命中 {cache_stats.embed_hit} / 未命中 {cache_stats.embed_miss}\n"
        f"  Answer 缓存:    命中 {cache_stats.answer_hit} / 未命中 {cache_stats.answer_miss}\n"
        f"  Semantic 缓存:  命中 {cache_stats.semantic_hit} / 未命中 {cache_stats.semantic_miss}\n"
        f"  总命中率:       {cache_stats.hit_rate:.1f}%"
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
