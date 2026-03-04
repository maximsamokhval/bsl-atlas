# Data Model: DeepSeek LLM Provider Integration

**Feature**: 001-deepseek-llm-provider  
**Date**: 2026-03-04  
**Purpose**: Определить сущности данных, их атрибуты и отношения, возникающие при добавлении поддержки DeepSeek API и конфигурируемых провайдеров эмбеддингов.

## Сущности (Entities)

### LLMProvider (Протокол)

Абстракция провайдера языковых моделей. Не является хранимой сущностью, но определяет контракт, который должны реализовывать все конкретные провайдеры.

- **Тип**: Protocol (Python `typing.Protocol`)
- **Методы**:
  - `generate(prompt: str, **kwargs) -> str` – синхронная генерация текста.
  - `async_generate(prompt: str, **kwargs) -> Awaitable[str]` – асинхронная версия (опционально).
- **Атрибуты**: нет (состояние управляется конкретной реализацией).
- **Валидация**: Контракт требует, чтобы реализация возвращала строку (ответ LLM) или вызывала исключение при ошибке.

### DeepSeekProvider

Конкретная реализация `LLMProvider` для работы с DeepSeek API.

- **Тип**: Класс (реализует `LLMProvider`).
- **Атрибуты**:
  - `api_key: str` – ключ API (берётся из конфигурации).
  - `base_url: str = "https://api.deepseek.com"` – базовый URL API.
  - `model: str = "deepseek-chat"` – идентификатор модели (по умолчанию).
  - `timeout: float = 30.0` – таймаут запроса в секундах.
  - `max_tokens: int | None = None` – максимальное количество токенов в ответе.
  - `temperature: float = 0.7` – температура генерации.
- **Методы**: реализует `generate` и `async_generate`, выполняющие HTTP‑запрос к `/v1/chat/completions`.
- **Валидация**: Проверяет наличие `api_key`, корректность `base_url` (должен начинаться с `https://`).

### EmbeddingProvider (расширение существующего протокола)

Существующий протокол из `src/indexer/embeddings.py` расширяется поддержкой значения `"deepinfra"`.

- **Тип**: Protocol (уже определён).
- **Методы**: `embed_documents(texts: list[str]) -> list[list[float]]`, `embed_query(text: str) -> list[float]`.
- **Новое значение провайдера**: `"deepinfra"` – соответствует API Deepinfra (Qwen3‑Embedding‑4B).
- **Атрибуты для deepinfra**:
  - `api_key: str | None` – ключ Deepinfra (может быть необязательным для бесплатного тарифа).
  - `base_url: str = "https://api.deepinfra.com/v1/inference"`.
  - `model: str = "Qwen/Qwen3-Embedding-4B"`.

### Configuration (расширение существующего класса)

Класс `Config` из `src/config.py` дополняется новыми полями.

- **Тип**: Pydantic‑модель (или dataclass).
- **Новые поля**:
  - `deepseek_api_key: str | None = None` – ключ DeepSeek API (берётся из `DEEPSEEK_API_KEY`).
  - `deepseek_base_url: str = "https://api.deepseek.com"` – из `DEEPSEEK_BASE_URL` (опционально).
  - `embedding_provider: Literal["ollama", "openai", "openrouter", "cohere", "jina", "local", "deepinfra"]` – расширенный перечень допустимых провайдеров.
- **Валидация**: Если `embedding_provider == "deepinfra"`, проверять наличие `DEEPINFRA_API_KEY` (если требуется). Если `deepseek_api_key` отсутствует, но попытка использовать DeepSeekProvider – ошибка конфигурации.

### LLMRequest (Data Transfer Object)

Входные данные для запроса к LLM. Не хранится, используется как DTO.

- **Тип**: Pydantic‑модель (или dataclass).
- **Атрибуты**:
  - `prompt: str` – текст запроса.
  - `model: str | None = None` – переопределение модели.
  - `temperature: float | None = None`.
  - `max_tokens: int | None = None`.
  - `stream: bool = False` – поддерживается ли потоковый ответ.
- **Валидация**: `prompt` не должен быть пустым, `temperature` в диапазоне [0, 2].

### LLMResponse (Data Transfer Object)

Ответ от LLM.

- **Тип**: Pydantic‑модель.
- **Атрибуты**:
  - `text: str` – сгенерированный текст.
  - `model: str` – использованная модель.
  - `usage: dict[str, int] | None` – статистика использования токенов.
  - `finish_reason: str | None` – причина завершения.
- **Валидация**: `text` не может быть `None`.

## Отношения (Relationships)

- `DeepSeekProvider` реализует `LLMProvider`.
- `Configuration` содержит настройки для `DeepSeekProvider` и `EmbeddingProvider`.
- `LLMRequest` передаётся в `LLMProvider.generate()`.
- `LLMResponse` возвращается из `LLMProvider.generate()`.

## Состояния (State Transitions)

Неприменимо – провайдеры не имеют сложного жизненного цикла, они создаются при инициализации приложения и живут до его завершения.

## Миграции (Migrations)

Не требуются, так как:
1. Новые поля конфигурации добавляются со значениями по умолчанию (`None`), что обратно совместимо.
2. Расширение перечня `embedding_provider` не ломает существующий код – значение `"deepinfra"` просто становится допустимым.

## Примечания

- Все новые сущности должны быть покрыты тестами (TDD).
- Документация (docstrings) на русском языке обязательна.
- Импорт существующих сущностей (`EmbeddingProvider`, `Config`) должен осуществляться без изменения их исходного кода (Brownfield).