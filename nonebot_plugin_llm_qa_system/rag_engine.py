"""nonebot_plugin_llm_qa_system - RAG 引擎（Ollama 嵌入 + 语义搜索 + LLM 生成）"""

import json
import math
from typing import Any

from cachetools import TTLCache
from nonebot import logger

from .config import Config
from src.plugins.gpu_worker.client import GpuWorkerClient


class RAGEngine:
    """基于 Ollama 的 RAG 引擎，提供嵌入、检索、问答能力。"""

    def __init__(
        self,
        config: Config,
        gpu_client: GpuWorkerClient | None = None,
    ) -> None:
        self.config = config
        self._gpu_client = gpu_client
        self._owns_client = gpu_client is None
        self._embed_cache: TTLCache = TTLCache(maxsize=5000, ttl=86400)  # 进程内嵌入缓存

    # ==================== 嵌入 ====================

    async def embed(self, text: str) -> list[float]:
        """调用 GPU Worker 生成文本嵌入向量。

        内置进程级内存缓存，同次运行中重复文本直接返回缓存结果。
        """
        if text in self._embed_cache:
            return self._embed_cache[text]

        client = await self._get_client()
        result = await client.ollama_embed(text, model=self.config.llm_qa_embed_model)
        if result is None:
            raise RuntimeError("嵌入生成失败，GPU Worker 返回空结果")
        self._embed_cache[text] = result
        return result

    # ==================== 相似度 ====================

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ==================== 检索 ====================

    async def retrieve(
        self,
        query: str,
        entries: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """检索与查询最相关的知识条目。

        Args:
            query: 用户查询文本。
            entries: 知识条目列表，每项含 id, title, content, embedding。
            top_k: 返回条数，默认使用配置值。

        Returns:
            按相似度降序排列的条目列表。

        Raises:
            RuntimeError: 嵌入生成失败时抛出。
        """
        if not entries:
            return []

        top_k = top_k or self.config.llm_qa_top_k
        query_emb = await self.embed(query)
        if not query_emb:
            raise RuntimeError("查询嵌入生成失败，无法执行检索")

        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in entries:
            emb = json.loads(entry.get("embedding", "[]") or "[]")
            if not emb:
                # 嵌入为空时自动重新生成
                embed_text = f"{entry.get('title', '')}\n{entry.get('content', '')}"
                emb = await self.embed(embed_text)
                entry["embedding"] = json.dumps(emb)
            score = self.cosine_similarity(query_emb, emb)
            scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        min_score = self.config.llm_qa_min_score
        return [entry for score, entry in scored[:top_k] if score >= min_score]

    # ==================== 问答 ====================

    async def ask(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        max_context_chars: int = 6000,
    ) -> str:
        """调用 Ollama 生成回答。

        Args:
            query: 用户问题。
            context_chunks: 检索到的相关条目。
            max_context_chars: 上下文最大字符数，超出时逐个截断条目内容。

        Returns:
            LLM 生成的回答文本。
        """
        # 构建上下文文本（带长度限制）
        context_parts: list[str] = []
        current_len = 0
        for i, chunk in enumerate(context_chunks, 1):
            title = chunk.get("title", f"文档{i}")
            content = chunk.get("content", "")
            part = f"[{i}] {title}\n{content}"

            remaining = max_context_chars - current_len
            if remaining <= 0:
                break
            if len(part) > remaining:
                part = part[:max(remaining - 30, 0)] + "\n...[内容过长，已截断]"
            context_parts.append(part)
            current_len += len(part)

        context_text = "\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": self.config.llm_qa_system_prompt},
        ]
        if context_text:
            messages.append({
                "role": "user",
                "content": (
                    f"请根据以下参考信息回答问题。\n\n"
                    f"参考信息：\n{context_text}\n\n"
                    f"问题：{query}"
                ),
            })
        else:
            messages.append({"role": "user", "content": query})

        try:
            client = await self._get_client()
            result = await client.ollama_chat(messages, model=self.config.llm_qa_chat_model)
            if result is None:
                raise RuntimeError("回答生成失败，GPU Worker 返回空结果")
            return result
        except Exception as e:
            logger.error(f"llm_qa: LLM 调用失败: {e}")
            return f"抱歉，调用语言模型时出错：{e}"

    async def _get_client(self) -> GpuWorkerClient:
        """获取 GPU Worker 客户端，按需延迟初始化。"""
        if self._gpu_client is None:
            self._gpu_client = GpuWorkerClient()
            self._owns_client = True
        return self._gpu_client

    async def close(self) -> None:
        """关闭 GPU Worker 客户端（仅当由本引擎创建时）。"""
        if self._owns_client and self._gpu_client is not None:
            await self._gpu_client.close()
            self._gpu_client = None
