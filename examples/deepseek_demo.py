#!/usr/bin/env python3
"""
Пример использования новых провайдеров DeepSeek (LLM) и Deepinfra (эмбеддинги).

Перед запуском убедитесь, что установлены зависимости проекта и заданы переменные окружения:

```bash
export DEEPSEEK_API_KEY="sk-..."           # ключ DeepSeek API
export EMBEDDING_PROVIDER="deepinfra"      # или "ollama"
export DEEPINFRA_API_KEY=""                # опционально (можно оставить пустым)
```

Запуск:
    python examples/deepseek_demo.py
"""

import os
import asyncio
import sys

# Добавляем корневую директорию проекта в путь, чтобы импортировать модули
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo_llm():
    """Синхронная генерация текста через DeepSeek API."""
    from src.llm.factory import create_llm_provider

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  DEEPSEEK_API_KEY не задан. Пропускаем демо LLM.")
        return

    print("🧠 Демо DeepSeek LLM провайдера (синхронный запрос)...")
    provider = create_llm_provider(
        provider="deepseek",
        api_key=api_key,
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=100,
    )

    prompt = "Напиши короткое приветствие на русском языке для пользователя BSL Atlas."
    try:
        response = provider.generate(prompt)
        print(f"📤 Запрос: {prompt}")
        print(f"📥 Ответ: {response}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


async def demo_llm_async():
    """Асинхронная генерация текста через DeepSeek API."""
    from src.llm.factory import create_llm_provider

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return

    print("\n🧠 Демо DeepSeek LLM провайдера (асинхронный запрос)...")
    provider = create_llm_provider(
        provider="deepseek",
        api_key=api_key,
    )

    prompt = "Объясни в одном предложении, что такое векторный поиск."
    try:
        response = await provider.async_generate(prompt)
        print(f"📤 Запрос: {prompt}")
        print(f"📥 Ответ: {response}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


def demo_embeddings():
    """Генерация эмбеддингов через Deepinfra (или другой провайдер)."""
    from src.indexer.embeddings import create_embedding_provider

    provider_name = os.getenv("EMBEDDING_PROVIDER", "ollama")
    api_key = os.getenv("DEEPINFRA_API_KEY")

    print(f"\n🔤 Демо провайдера эмбеддингов '{provider_name}'...")

    try:
        embedding_provider = create_embedding_provider(
            provider=provider_name,
            api_key=api_key,
            model="Qwen/Qwen3-Embedding-4B" if provider_name == "deepinfra" else None,
        )
    except Exception as e:
        print(f"❌ Не удалось создать провайдер эмбеддингов: {e}")
        return

    texts = [
        "Модуль управления справочником Номенклатура",
        "Общий модуль учета товаров",
        "Документ ПоступлениеТоваровУслуг",
    ]

    try:
        embeddings = embedding_provider.embed_documents(texts)
        print(f"✅ Успешно сгенерировано {len(embeddings)} эмбеддингов.")
        if embeddings and embeddings[0]:
            print(f"   Размерность первого эмбеддинга: {len(embeddings[0])}")
    except Exception as e:
        print(f"❌ Ошибка при генерации эмбеддингов: {e}")


def main():
    print("🚀 Запуск демонстрации новых провайдеров (DeepSeek LLM + Deepinfra embeddings)")
    print("=" * 70)

    demo_llm()
    asyncio.run(demo_llm_async())
    demo_embeddings()

    print("\n✅ Демо завершено.")


if __name__ == "__main__":
    main()
