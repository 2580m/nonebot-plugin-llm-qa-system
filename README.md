# nonebot-plugin-llm-qa-system

基于本地 Ollama 大模型 + RAG（检索增强生成）+ 多级缓存的 NoneBot2 智能问答插件。

## 功能

- **问答**：基于知识库的内容，利用 LLM 生成回答
- **添加知识**：向知识库添加条目，自动生成语义嵌入向量
- **语义搜索**：通过余弦相似度检索相关知识
- **知识管理**：列出、删除、清空知识条目
- **多级缓存**：PromptCache → SemanticCache → AnswerCache，逐层拦截降低 CPU 推理开销
- **缓存管理**：查看缓存状态和命中统计，手动清空缓存

## 安装

```bash
pip install nonebot-plugin-llm-qa-system
```

或者将本插件目录复制到项目的 `src/plugins/` 下，然后在 `pyproject.toml` 中注册：

```toml
[tool.nonebot]
plugins = ["nonebot_plugin_llm_qa_system"]
```

## 前置依赖

- [Ollama](https://ollama.com/) 本地运行
- 所需的模型（首次使用前需拉取）：

```bash
ollama pull qwen3:1.7b         # 对话模型（默认）
ollama pull bge-m3              # 嵌入模型（推荐）
```

## 配置

在项目 `.env` 文件中添加以下配置项：

```env
# —— 数据库（必须）——
# 默认使用 SQLite，通过 nonebot-plugin-orm 管理
# 可自定义路径，确保目录已创建
SQLALCHEMY_DATABASE_URL=sqlite+aiosqlite:///path/to/data/llm_qa.db

# —— Ollama 服务 ——
# Ollama 服务地址（默认值 http://localhost:11434）
llm_qa_ollama_host=http://localhost:11434

# 对话模型名称（默认值 qwen3:1.7b）
llm_qa_chat_model=qwen3:1.7b

# 嵌入模型名称（默认值 nomic-embed-text，推荐 bge-m3）
llm_qa_embed_model=bge-m3

# —— RAG 检索 ——
# RAG 检索返回的最大相关文档数（默认值 3）
llm_qa_top_k=3

# 余弦相似度最低阈值，低于该值的结果不返回（默认值 0.3）
llm_qa_min_score=0.3

# —— 语义缓存 ——
# 语义缓存命中阈值，余弦相似度高于此值时直接返回缓存答案（默认值 0.97）
semantic_cache_threshold=0.97

# 语义缓存候选数，每次查询从最近 N 条缓存中匹配（默认值 200）
semantic_cache_max_candidates=200
```

## 使用

插件目前仅支持 **QQ 群聊**，所有命令通过群消息触发。

| 命令 | 权限 | 说明 |
|------|------|------|
| `问答 <问题>` | 群员 | 基于知识库回答用户问题 |
| `添加知识 <标题> <内容>` | 群员 | 向知识库添加条目 |
| `删除知识 <id>` | SUPERUSER | 删除指定条目 |
| `列出知识` | 群员 | 列出知识库所有条目 |
| `搜索知识 <关键词>` | 群员 | 语义搜索知识库 |
| `清空知识` | SUPERUSER | 清空全部条目（需确认） |
| `缓存状态` | SUPERUSER | 查看各级缓存大小和命中统计 |
| `清空缓存` | SUPERUSER | 清空内存中的问答缓存（需确认） |

### 示例

```
问答 RHEL 是什么？
添加知识 Docker安装 使用以下命令安装 Docker...
删除知识 3
列出知识
搜索知识 防火墙配置
清空知识 确认
缓存状态
```

## 缓存架构

```
用户问题
    ↓
PromptCache（LRU 精确匹配，内存）
    ↓
SemanticCache（向量相似度 ≥ 0.97，SQLite）
    ↓
RAG 检索
    ↓
AnswerCache（内容指纹精确匹配，SQLite）
    ↓
LLM 生成
    ↓
写入各级缓存
```

### 各级缓存说明

| 缓存层 | 存储位置 | 匹配方式 | 作用 |
|--------|---------|---------|------|
| PromptCache | 进程内存（LRU，max=100） | 归一化后的精确文本匹配 | 同一措辞的问题秒回 |
| SemanticCache | SQLite | query 向量余弦相似度 ≥ 阈值 | 语义相近的问题跳过 RAG + LLM |
| AnswerCache | SQLite | SHA256(knowledge_version + system_prompt + chat_model + 内容指纹 + 归一化问题) | 相同上下文 + 问题精确命中 |
| EmbeddingCache | SQLite + 进程 TTLCache | 文本 + 模型名精确匹配 | 避免重复调用 Ollama 嵌入 API |

### 缓存失效策略

- **知识变更**（添加/删除/清空）：`knowledge_version` 递增，SemanticCache 和 AnswerCache 的旧版本条目自动隔离，不删除
- **Prompt 修改**：缓存键包含 `system_prompt`，prompt 内容变化自动 Miss
- **切换模型**：缓存键包含 `chat_model`，模型变化自动 Miss
- **语义缓存清理**：自动删除 30 天前的条目或版本差超过 5 的条目

### 缓存统计

`缓存状态` 命令输出示例：

```
📊 缓存状态：
  知识库版本：        5
  问答缓存（LRU 精确匹配）：3 条
  回答缓存（哈希匹配）：12 条
  嵌入缓存（持久化）：  20 条
  嵌入缓存（进程级）：  8 条

📈 缓存命中统计：
  Embedding 缓存: 命中 42 / 未命中 5
  Answer 缓存:    命中 15 / 未命中 8
  Semantic 缓存:  命中 6 / 未命中 17
  总命中率:       63.3%
```

## 工作原理

1. 用户发送 `问答 <问题>`
2. **PromptCache** 拦截：归一化后的问题精确匹配到历史缓存？直接返回
3. **SemanticCache** 拦截：计算问题向量，与最近 N 条缓存做余弦相似度，最高分 ≥ 阈值？返回缓存答案（跳过 RAG + LLM）
4. **加载知识库**：从 SQLite 加载全部知识条目
5. **RAG 检索**：计算问题与每条知识的余弦相似度，返回 top_k 中高于 min_score 的条目
6. **AnswerCache** 拦截：基于知识内容指纹 + 知识库版本 + 系统提示 + 模型名 + 归一化问题的 SHA256 哈希精确匹配？返回
7. **LLM 生成**：拼接 Prompt 发送给 Ollama 对话模型
8. **写入缓存**：同步写入 PromptCache + AnswerCache + SemanticCache

## 兼容性

插件自动兼容不同版本的 Ollama 嵌入 API：
- 优先尝试新版 `/api/embed`（Ollama >= 0.1.24）
- 失败时自动降级到旧版 `/api/embeddings`

## 依赖

- `nonebot2>=2.0.0`
- `nonebot-adapter-onebot>=2.0.0`
- `nonebot-plugin-orm>=1.0.0`
- `httpx>=0.24.0`
- `jieba_next>=1.0.0`（查询文本归一化分词）
- `cachetools>=5.3.0`（进程级 TTL 缓存）
- `Ollama`（外部服务）

## 许可证

MIT
