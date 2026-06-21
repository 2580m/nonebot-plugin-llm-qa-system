"""nonebot_plugin_llm_qa_system - RAG 引擎（Ollama 嵌入 + 语义搜索 + LLM 生成）"""

import json
import math
from typing import Any

import httpx
from nonebot import logger

from .config import Config


class RAGEngine:
    """基于 Ollama 的 RAG 引擎，提供嵌入、检索、问答能力。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._http = httpx.AsyncClient(
            base_url=config.llm_qa_ollama_host,
            timeout=60,
        )
        self._embed_api_ver: int | None = None  # 1 = /api/embeddings, 2 = /api/embed

    # ==================== 嵌入 ====================

    async def embed(self, text: str) -> list[float]:
        """调用 Ollama 生成文本嵌入向量。

        优先尝试新版 /api/embed API，失败时回退到旧版 /api/embeddings。
        """
        if self._embed_api_ver == 2 or self._embed_api_ver is None:
            try:
                return await self._embed_v2(text)
            except Exception as e:
                if self._embed_api_ver == 2:
                    raise
                logger.warning(f"llm_qa: /api/embed 失败，尝试 /api/embeddings: {e}")

        return await self._embed_v1(text)

    async def _embed_v2(self, text: str) -> list[float]:
        """新版 Ollama 嵌入 API (>=0.1.24)"""
        resp = await self._http.post(
            "/api/embed",
            json={
                "model": self.config.llm_qa_embed_model,
                "input": text,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        # 新版返回 embeddings: list[list[float]]
        embeddings = data.get("embeddings")
        if embeddings and isinstance(embeddings, list) and len(embeddings) > 0:
            self._embed_api_ver = 2
            return embeddings[0]

        # 某些版本可能返回 embedding: list[float]
        single = data.get("embedding")
        if single and isinstance(single, list):
            self._embed_api_ver = 2
            return single

        raise RuntimeError(f"无法解析 /api/embed 响应: {data.keys()}")

    async def _embed_v1(self, text: str) -> list[float]:
        """旧版 Ollama 嵌入 API"""
        resp = await self._http.post(
            "/api/embeddings",
            json={
                "model": self.config.llm_qa_embed_model,
                "prompt": text,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        embedding = data.get("embedding")
        if embedding and isinstance(embedding, list):
            self._embed_api_ver = 1
            return embedding

        raise RuntimeError(f"无法解析 /api/embeddings 响应: {data.keys()}")

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
                continue
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
            resp = await self._http.post(
                "/api/chat",
                json={
                    "model": self.config.llm_qa_chat_model,
                    "messages": messages,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "抱歉，我没有得到有效的回答。")
        except Exception as e:
            logger.error(f"llm_qa: LLM 调用失败: {e}")
            return f"抱歉，调用语言模型时出错：{e}"

    async def close(self) -> None:
        """关闭 HTTP 客户端。"""
        await self._http.aclose()
