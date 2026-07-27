# BSL Atlas

[![Docker Hub](https://img.shields.io/docker/v/armankudaibergenov/bsl-atlas?label=Docker%20Hub&logo=docker)](https://hub.docker.com/r/armankudaibergenov/bsl-atlas)

Публичный MCP-сервер для быстрой индексации и поиска по исходникам 1С. Работает с XML/BSL-выгрузкой конфигурации или расширения и отдает структурный и, при необходимости, семантический поиск для AI-ассистентов.

## Что умеет

- искать функции, процедуры и модули через SQLite/FTS
- искать объекты метаданных, реквизиты и связи
- строить контекст по вызовам и структуре модулей
- работать в `fast` режиме без внешних embedding API
- переиндексировать проект после новой выгрузки

## Режимы

| Режим | Что дает | Что нужно |
|------|----------|------------|
| `fast` | быстрый структурный поиск | Docker и выгруженные исходники 1С |
| `full` | структурный + семантический поиск | Docker, исходники и embedding backend/API key |

`fast` — основной и рекомендуемый стартовый режим.

## Важно: mount исходников обязателен

Если вы запускаете `bsl-atlas` в Docker, контейнер обязан видеть реальные исходники проекта через bind mount `SOURCE_PATH -> /data/source`.

- `SOURCE_PATH` нужен для индексации файлов
- если bind mount настроен неверно, `/data/source` внутри контейнера может существовать, но будет пустым
- в этом случае Atlas честно сообщит, что каталог исходников пустой

Это отдельная тема от RLM: Atlas читает файлы проекта напрямую, поэтому без source mount индексировать нечего.

## Быстрый старт

### 1. Выгрузите исходники 1С

В конфигураторе используйте `Конфигурация -> Выгрузить конфигурацию в файлы` и укажите пустой каталог.

### 2. Скачайте конфиги

```bash
curl -O https://raw.githubusercontent.com/Arman-Kudaibergenov/bsl-atlas/master/docker-compose.yml
curl -O https://raw.githubusercontent.com/Arman-Kudaibergenov/bsl-atlas/master/.env.example
cp .env.example .env
```

### 3. Заполните `.env`

```env
SOURCE_PATH=C:\bsl-src
INDEXING_MODE=fast
```

Для `full` режима дополнительно укажите embedding provider и нужные ключи.

### 4. Запустите контейнер

```bash
docker compose up -d
```

### 5. Подключите MCP в Claude

Добавьте в `claude_desktop_config.json` или в `.mcp.json` проекта:

```json
{
  "mcpServers": {
    "bsl-atlas": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

## Windows: что важно

Docker Desktop на Windows часто ломается на путях с пробелами или кириллицей. Если реальный путь выглядит как `C:\1С\Выгрузки\МояКонфигурация`, лучше сначала сделать ASCII-алиас.

```powershell
cmd /c mklink /D C:\bsl-src "C:\1С\Выгрузки\МояКонфигурация"
```

После этого в `.env` используйте:

```env
SOURCE_PATH=C:\bsl-src
```

Если Atlas пишет, что `SOURCE_PATH` пустой, проблема почти всегда в bind mount, а не в самом приложении.

## Поддерживаемые структуры

Каталог исходников может выглядеть так:

```text
SOURCE_PATH/
  cf/
    Catalogs/
    Documents/
    CommonModules/
```

или так:

```text
SOURCE_PATH/
  Catalogs/
  Documents/
  CommonModules/
```

или так:

```text
SOURCE_PATH/
  cfe/
    MyExtension/
      Catalogs/
      CommonModules/
```

## Основные инструменты

- `search_function(name)` — найти функцию или процедуру по имени
- `get_module_functions(path)` — список функций модуля
- `get_function_context(name)` — контекст вызовов
- `metadatasearch(query)` — поиск по объектам метаданных
- `get_object_details(full_name)` — структура объекта
- `codesearch(query)` — семантический поиск в `full` режиме
- `helpsearch(query)` — поиск по help/knowledge слою в `full` режиме
- `reindex(force_chromadb)` — переиндексация после изменений
- `stats()` — статистика индекса

## Переиндексация

После новой выгрузки исходников:

```bash
curl -X POST http://localhost:8000/reindex
```

## Embedding defaults

- рекомендуемое семейство: `qwen3-embedding-8b`
- OpenRouter: `qwen/qwen3-embedding-8b`
- Ollama: `qwen3-embedding:8b`

## LLM-стикеры функций

Стикер — короткое русское описание назначения BSL-функции: что она делает,
что возвращает и какой имеет побочный эффект. Он помогает найти функцию по
смыслу, когда точное имя неизвестно. Стикеры хранятся в
`symbols.doc_generated` и добавляются к карточке функции при следующей
векторной индексации.

Генерировать стикеры стоит только для функций без хорошего `//`-комментария:
экспортных методов, обработчиков, точек входа и центральных узлов графа.
Формы и очевидные вспомогательные функции обычно не окупают стоимость генерации.

### Пример генерации через Claude Haiku

Актуальный закреплённый ID модели — `claude-haiku-4-5-20251001`.

```python
import json
import os

from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
function = {
    "id": 42,
    "name": "ОбработкаПроведения",
    "module": "Документы/ЗаказКлиента/Ext/ObjectModule.bsl",
    "signature": "Процедура ОбработкаПроведения(Отказ, РежимПроведения)",
    "body": "Движения.Взаиморасчеты.Записать();",
}

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    max_tokens=200,
    system=(
        "Создай один русский стикер для поиска по BSL-коду. "
        "Укажи назначение, возвращаемое значение и побочный эффект, если они видны. "
        "Не выдумывай объекты и бизнес-смысл. Не более 160 символов. "
        "Верни только JSON: {\"id\": число, \"description\": \"текст\"}."
    ),
    messages=[{
        "role": "user",
        "content": json.dumps(function, ensure_ascii=False),
    }],
)
print(response.content[0].text)
```

Ожидаемый результат:

```json
{"id": 42, "description": "Проводит заказ клиента и записывает движения по взаиморасчётам."}
```

Для массовой генерации используйте Anthropic Message Batches: пакетный API
дешевле обычных синхронных запросов. Сохраняйте результат как JSON
`{symbol_id: description}`:

```json
{"42": "Проводит заказ клиента и записывает движения по взаиморасчётам."}
```

Импорт:

```bash
python scripts/import_generated_docs.py descriptions.json --db data/bsl_index.db
```

После импорта выполните `reindex_changed`, чтобы обновить векторы затронутых
модулей. Скрипт не принимает и не хранит токены LLM.
