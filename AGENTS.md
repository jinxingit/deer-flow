# 仓库贡献指南

## 项目结构与模块组织
- **后端**：核心 Python 代码位于 `src/`，重点目录包括 `server/`（FastAPI 接口）、`graph/`（LangGraph 工作流）、`agents/`（智能体模版）以及 `tools/`（工具集成）。会话持久化相关代码集中在 `src/server/session/`。
- **前端**：单页应用资源位于 `web/`；业务路由在 `web/src/app/`，共享逻辑在 `web/src/core/`，基础组件在 `web/src/components/`。
- **资源与文档**：静态资源保存在 `assets/`，参考文档与规划（如 `docs/session_management_plan.md`）位于 `docs/`。测试按粒度分布于 `tests/unit/` 与 `tests/integration/`。

## 构建、测试与开发命令
- `uv sync`：安装 `pyproject.toml` 声明的 Python 依赖。
- `uv run uvicorn src.server.app:app --reload`：启动 FastAPI 后端服务。
- `pnpm install` 后执行 `pnpm dev`：安装前端依赖并运行 Next.js 开发服务器。
- `make serve`：通过脚本同时拉起前后端服务。
- `uv run pytest`：运行 Python 测试套件（默认开启覆盖率统计）。

## 代码风格与命名约定
- Python 版本为 3.12，统一四空格缩进，Ruff 限制 88 字符行宽；函数与变量使用 snake_case，类使用 PascalCase，模块名保持小写。
- 新增公共函数需补充类型标注与中文 Docstring，可复用的提示词统一置于 `src/prompts/`。
- 前端遵循 TypeScript/React 函数式组件风格，共享 Hook 归档在 `web/src/core/hooks/`。
- **重要**：所有自动生成的文档与代码注释必须使用中文撰写，保持术语一致。

## 测试规范
- Pytest 会发现 `test_*.py` 文件；单元测试置于 `tests/unit/`，集成及 LangGraph 流程测试置于 `tests/integration/`。
- 异步测试使用 `pytest-asyncio`，涉及外部服务时需确保夹具清理 Postgres、Milvus、Mongo 等依赖。
- 项目覆盖率下限为 25%，新功能应提供针对性用例，如 `test_session_title_truncation`。
- 前端（若新增）推荐使用 Playwright 或 Vitest，并将测试放置于组件同级或 `__tests__/` 目录。

## 提交与合并请求规范
- 遵循约定式提交前缀（如 `feat:`、`fix:`、`docs:`、`refactor:`），后接精炼动词短语，例如 `feat: add sqlite-backed session management`。
- 提交 PR 前需通过 `uv run pytest`、`make lint`，若修改 UI 需额外执行 `make lint-frontend`。
- PR 描述应包含动机、影响范围、回归风险，并关联相关 Issue；界面变更需附截图或录屏，同时补充新增环境变量或配置步骤。

## 安全与配置提示
- 禁止提交真实凭证。可复制 `conf.yaml.example` 自定义配置，并将敏感信息写入 `.env`，在代码中通过 `get_str_env(...)` 读取。
- 部署时将 `SESSION_DB_PATH` 指向持久化目录，以保存多会话聊天数据。
