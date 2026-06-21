# nonebot-plugin-llm-qa-system

基于本地 Ollama 大模型 + RAG（检索增强生成）的 NoneBot2 智能问答插件。

## 功能

- **问答**：基于知识库的内容，利用 LLM 生成回答
- **添加知识**：向知识库添加条目，自动生成语义嵌入向量
- **语义搜索**：通过余弦相似度检索相关知识
- **知识管理**：列出、删除、清空知识条目

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
ollama pull nomic-embed-text   # 嵌入模型
```

## 配置

在项目 `.env` 文件中添加以下配置项：

```env
# —— 数据库（必须）——
# 默认使用 SQLite，通过 nonebot-plugin-orm 管理
# 可自定义路径，确保目录已创建
SQLALCHEMY_DATABASE_URL=sqlite+aiosqlite:///path/to/data/llm_qa.db

# —— 插件配置 ——
# Ollama 服务地址（默认值 http://localhost:11434）
llm_qa_ollama_host=http://localhost:11434

# 对话模型名称（默认值 qwen3:1.7b）
llm_qa_chat_model=qwen3:1.7b

# 嵌入模型名称（默认值 nomic-embed-text）
llm_qa_embed_model=nomic-embed-text

# RAG 检索返回的最大相关文档数（默认值 3）
llm_qa_top_k=3

# 余弦相似度最低阈值，低于该值的结果不返回（默认值 0.3）
llm_qa_min_score=0.3
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

### 示例

```
问答 RHEL 是什么？
添加知识 Docker安装 使用以下命令安装 Docker...
删除知识 3
列出知识
搜索知识 防火墙配置
清空知识 确认
```

## 工作原理

```
用户提问 → 嵌入查询向量 → 余弦相似度检索知识库 → 拼接上下文 → LLM 生成回答
                ↓                    ↑
          Ollama nomic-embed-text    知识库（SQLite + SQLAlchemy ORM）
                                     ↓
                               Ollama qwen3:1.7b
```

1. 用户发送 `问答 <问题>`
2. 插件从 SQLite 加载全部知识条目
3. 调用 Ollama 的嵌入 API 将问题转为向量
4. 计算所有条目的余弦相似度，返回 top_k 中高于 min_score 的条目
5. 拼接为 Prompt 发送给 Ollama 对话模型
6. 返回 LLM 生成的回答和参考来源

## 兼容性

插件自动兼容不同版本的 Ollama 嵌入 API：
- 优先尝试新版 `/api/embed`（Ollama >= 0.1.24）
- 失败时自动降级到旧版 `/api/embeddings`

## 依赖

- `nonebot2>=2.0.0`
- `nonebot-adapter-onebot>=2.0.0`
- `nonebot-plugin-orm>=1.0.0`
- `httpx>=0.24.0`
- `Ollama`（外部服务）

## 许可证

MIT
