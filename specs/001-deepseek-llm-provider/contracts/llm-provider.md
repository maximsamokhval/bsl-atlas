# Contract: LLM Provider Interface

**Feature**: 001-deepseek-llm-provider  
**Date**: 2026-03-04  
**Audience**: Разработчики, добавляющие новых LLM‑провайдеров; потребители LLM‑функциональности внутри проекта.

## Назначение

Определить стандартный интерфейс, которому должны соответствовать все реализации LLM‑провайдеров в проекте bsl‑atlas. Контракт гарантирует взаимозаменяемость провайдеров и упрощает тестирование.

## Контракт (Protocol)

```python
from typing import Protocol, runtime_checkable, Any

@runtime_checkable
class LLMProvider(Protocol):
    """Протокол провайдера языковых моделей."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Синхронно генерирует текст по заданному промпту.

        Args:
            prompt: Текст промпта.
            **kwargs: Дополнительные параметры (model, temperature, max_tokens и т.д.).

        Returns:
            Сгенерированный текст.

        Raises:
            LLMError: Если произошла ошибка API, сети или аутентификации.
        """
        ...

    async def async_generate(self, prompt: str, **kwargs: Any) -> str:
        """Асинхронная версия generate.

        Требуется, если провайдер поддерживает асинхронные вызовы.
        Если не поддерживается, можно возбудить NotImplementedError.
        """
        ...
```

## Обязательное поведение

1. **Идемпотентность**: Вызов `generate` с одинаковыми `prompt` и `kwargs` может возвращать разный текст (из‑за природы LLM), но не должен приводить к побочным эффектам (например, списанию токенов сверх одного запроса).
2. **Обработка ошибок**: Любая ошибка, связанная с сетью, API или валидацией, должна быть обёрнута в исключение `LLMError` (или его подкласс), определённое в модуле `llm.exceptions`.
3. **Логирование**: Каждый вызов должен логироваться через `loguru` на уровне `DEBUG` (без раскрытия секретных данных).
4. **Таймаут**: Провайдер должен соблюдать таймаут, заданный в конфигурации (по умолчанию 30 секунд). Превышение таймаутов должно вызывать `TimeoutError`.

## Расширяемые параметры (kwargs)

Провайдер должен поддерживать следующие стандартные параметры, передаваемые через `**kwargs`:

- `model: str | None` – идентификатор модели (переопределяет модель по умолчанию).
- `temperature: float | None` – температура генерации (0..2).
- `max_tokens: int | None` – максимальное количество токенов в ответе.
- `stream: bool = False` – флаг потокового ответа (если поддерживается).

Провайдер может игнорировать неподдерживаемые параметры, но должен логировать предупреждение.

## Контракт фабрики

```python
def create_llm_provider(
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Фабрика для создания экземпляра LLMProvider.

    Args:
        provider: Идентификатор провайдера ("deepseek", "openai", "ollama" и т.д.).
        api_key: Ключ API (если требуется).
        base_url: Базовый URL API (если отличается от стандартного).
        model: Модель по умолчанию.
        **kwargs: Дополнительные параметры, специфичные для провайдера.

    Returns:
        Реализацию LLMProvider.

    Raises:
        ValueError: Если передан неизвестный провайдер.
        ConfigurationError: Если отсутствует обязательный параметр (например, api_key).
    """
```

## Пример использования

```python
from llm.factory import create_llm_provider

provider = create_llm_provider(
    provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat",
    temperature=0.7,
)

response = provider.generate("Напиши hello world на BSL")
print(response)  # "Процедура HelloWorld()..."
```

## Контракт для тестирования

В тестах разрешается использовать моки, реализующие `LLMProvider`. Минимальный мок должен:

- Иметь метод `generate`, возвращающий предопределённый ответ.
- Записывать факт вызова для последующих проверок (`assert_called_with`).

Пример мока:

```python
from unittest.mock import Mock

mock_provider = Mock(spec=LLMProvider)
mock_provider.generate.return_value = "заглушенный ответ"
```

## Совместимость с существующей архитектурой

Новый контракт `LLMProvider` строится по образцу существующего `EmbeddingProvider`. Это обеспечивает единообразие и позволяет использовать знакомые паттерны (фабрика, протокол, dependency injection).

Все реализации должны быть добавлены в проект без изменения уже работающего кода (принцип Brownfield).