# Quick Start: DeepSeek LLM Provider Integration

**Feature**: 001-deepseek-llm-provider  
**Date**: 2026-03-04  
**Цель**: Позволить разработчику/оператору быстро настроить и использовать DeepSeek API и провайдер эмбеддингов Deepinfra в проекте bsl‑atlas.

## Предварительные требования

- Установленный Python 3.11+
- Установленные зависимости проекта (`pip install -e .`)
- Аккаунт на [DeepSeek Platform](https://platform.deepseek.com/) для получения API‑ключа
- (Опционально) Аккаунт на [Deepinfra](https://deepinfra.com/) для использования Qwen3‑Embedding‑4B (можно использовать без ключа в пределах бесплатного лимита)

## Шаг 1: Настройка переменных окружения

Создайте файл `.env` в корне проекта или экспортируйте переменные в оболочке:

```bash
# DeepSeek API
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"  # опционально, по умолчанию используется этот URL

# Embedding provider (выберите один)
export EMBEDDING_PROVIDER="ollama"           # использовать локальный Ollama
# или
export EMBEDDING_PROVIDER="deepinfra"        # использовать Deepinfra Qwen3‑Embedding‑4B
export DEEPINFRA_API_KEY="your‑deepinfra‑key" # опционально, если требуется

# Если используете Ollama, убедитесь что сервер запущен
# ollama run qwen3-embedding:4b
```

## Шаг 2: Проверка конфигурации

Запустите конфигурационный скрипт (если есть) или просто запустите приложение, чтобы убедиться, что переменные прочитаны корректно:

```bash
python -c "from src.config import Config; c = Config(); print(f'Embedding provider: {c.embedding_provider}')"
```

Ожидаемый вывод:
```
Embedding provider: deepinfra
```

## Шаг 3: Использование DeepSeek провайдера в коде

### Пример 1: Синхронная генерация текста

```python
import os
from llm.factory import create_llm_provider

# Создание провайдера
provider = create_llm_provider(
    provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat",  # можно опустить – будет использована модель по умолчанию
    temperature=0.7,
)

# Запрос к LLM
response = provider.generate("Объясни, что такое цикл в BSL на русском.")
print(response)
```

### Пример 2: Асинхронная генерация

```python
import asyncio
import os
from llm.factory import create_llm_provider

async def main():
    provider = create_llm_provider(
        provider="deepseek",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )
    response = await provider.async_generate("Напиши функцию сложения двух чисел на BSL.")
    print(response)

asyncio.run(main())
```

### Пример 3: Использование эмбеддингов через Deepinfra

```python
from src.indexer.embeddings import create_embedding_provider

# Создание провайдера эмбеддингов (автоматически выбирается из EMBEDDING_PROVIDER)
embedding_provider = create_embedding_provider(
    provider=os.getenv("EMBEDDING_PROVIDER", "ollama"),
    api_key=os.getenv("DEEPINFRA_API_KEY"),
    model="Qwen/Qwen3-Embedding-4B",  # для deepinfra
)

# Генерация эмбеддингов для документов
embeddings = embedding_provider.embed_documents(["Модуль управления справочником", "Общий модуль"])
print(f"Размерность эмбеддинга: {len(embeddings[0])}")
```

## Шаг 4: Интеграция с существующими компонентами

### Использование в MCP‑сервере

Если проект используется как MCP‑сервер, новый LLM‑провайдер автоматически становится доступным через инструменты MCP. Проверьте, что в `src/mcp/server.py` (или аналогичном) добавлена регистрация провайдера.

### Использование в CLI

Команды CLI, которые используют LLM (например, `bsl-atlas generate-doc`), будут автоматически использовать настроенный провайдер, если он указан в конфигурации.

## Шаг 5: Тестирование

Запустите тесты нового модуля, чтобы убедиться, что всё работает:

```bash
pytest tests/llm/ -v
pytest tests/indexer/ -v -k embedding
```

Все тесты должны проходить. Помните, что тесты используют моки реальных API, поэтому наличие интернета и API‑ключей не требуется.

## Устранение неполадок

### Ошибка «Missing API key»

Убедитесь, что переменная `DEEPSEEK_API_KEY` установлена и экспортирована в текущем окружении. Проверьте:

```bash
echo $DEEPSEEK_API_KEY
```

### Ошибка таймаута

Если запросы к DeepSeek занимают больше 30 секунд, увеличьте таймаут через параметр `timeout`:

```python
provider = create_llm_provider(..., timeout=60.0)
```

### Ошибка «Unsupported provider»

Убедитесь, что значение `EMBEDDING_PROVIDER` равно `"ollama"` или `"deepinfra"`. Регистр важен.

### Deepinfra возвращает 429 (Too Many Requests)

Бесплатный тариф Deepinfra имеет ограничения по количеству запросов. Перейдите на платный тариф или временно переключитесь на Ollama.

## Дальнейшие шаги

- Ознакомьтесь с полной спецификацией: [`spec.md`](spec.md)
- Изучите детали реализации в [`plan.md`](plan.md) и [`data-model.md`](data-model.md)
- Добавьте своих LLM‑провайдеров, следуя контракту [`contracts/llm-provider.md`](contracts/llm-provider.md)

**Готово!** Теперь вы можете использовать DeepSeek для генерации текста и выбирать между Ollama и Deepinfra для создания эмбеддингов.