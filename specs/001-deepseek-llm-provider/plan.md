# Implementation Plan: DeepSeek LLM Provider Integration

**Branch**: `001-deepseek-llm-provider` | **Date**: 2026-03-04 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-deepseek-llm-provider/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Добавить поддержку DeepSeek API как провайдера LLM для генерации текста и обеспечить конфигурируемый выбор провайдера эмбеддингов (Ollama / Deepinfra Qwen). Реализация должна следовать существующей абстракции провайдеров и соблюдать три ключевых ограничения конституции (Brownfield, TDD, стандарты качества).

## Technical Context

**Language/Version**: Python 3.11+ (согласно pyproject.toml)  
**Primary Dependencies**: httpx, pydantic, loguru, chromadb, openai (клиент), ollama (для локальных эмбеддингов)  
**Storage**: SQLite (через SQLAlchemy 2.0 async), ChromaDB (векторное хранилище)  
**Testing**: pytest, pytest‑mock, responses  
**Target Platform**: Linux/macOS (серверное приложение, возможно развертывание в Docker)  
**Project Type**: библиотека/CLI‑утилита с MCP‑сервером  
**Performance Goals**: обработка запросов к DeepSeek API в пределах таймаута 30 сек, индексация эмбеддингов без деградации производительности  
**Constraints**: обязательное соблюдение Brownfield (не менять существующий код), TDD, линтер (ruff), типизация (mypy), безопасность (bandit)  
**Scale/Scope**: проект рассчитан на обработку сотен BSL‑модулей, десятки параллельных запросов к LLM API

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

1. **Brownfield**: реализация только добавляет новый код, не изменяет существующие файлы оригинального репозитория. **PASS**
2. **TDD**: все новые модули будут сопровождаться тестами, написанными до реализации. **PASS**
3. **Стандарты качества**: линтер (ruff), типизация (mypy), тесты (pytest), безопасность (bandit) обязательны перед коммитом. **PASS**
4. **Зависимости**: добавление новых зависимостей требует обоснования в спеке — в spec.md указана необходимость интеграции с DeepSeek API и Deepinfra, что оправдывает добавление библиотек httpx и, возможно, openai (если ещё не используется). **PASS**
5. **Безопасность**: API‑ключи только через переменные окружения, реальные API не вызываются в тестах. **PASS**

Все гейты пройдены, нарушений нет.

## Project Structure

### Documentation (this feature)

```text
specs/001-deepseek-llm-provider/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/
├── llm/                         # Новый модуль для LLM‑провайдеров
│   ├── __init__.py
│   ├── base.py                  # Абстракция LLMProvider
│   ├── deepseek.py              # Реализация DeepSeekProvider
│   └── factory.py               # Фабрика create_llm_provider (аналогично create_embedding_provider)
├── indexer/
│   ├── embeddings.py            # Расширение для поддержки deepinfra
│   └── ... (остальное без изменений)
├── config.py                    # Добавление конфигурационных переменных DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, EMBEDDING_PROVIDER=deepinfra
└── main.py                      # Интеграция LLM‑провайдера в существующие потоки (опционально)

tests/
├── llm/
│   ├── test_base.py
│   ├── test_deepseek.py         # Мокирование API, TDD
│   └── test_factory.py
├── indexer/
│   └── test_embeddings_deepinfra.py
└── integration/
    └── test_llm_integration.py
```

**Structure Decision**: Выбрана Option 1 (единый проект) с добавлением модуля `src/llm/`. Существующая структура `src/indexer/`, `src/search/`, `src/storage/` остаётся без изменений. Новый код следует тем же паттернам (протоколы, фабрики), что и embedding providers.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

Нарушений нет, таблица не требуется.

---

