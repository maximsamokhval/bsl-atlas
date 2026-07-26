"""Load generated function descriptions into an Atlas SQLite index.

Input is a JSON object mapping the immutable ``symbols.id`` to a short Russian
description, for example ``{"42": "Проводит документ и формирует движения."}``.
Run a code reindex afterwards so the changed cards are embedded into ChromaDB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_descriptions(path: Path) -> dict[int, str]:
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("JSON должен быть объектом вида {symbol_id: description}")

    descriptions: dict[int, str] = {}
    for raw_id, description in raw.items():
        try:
            symbol_id = int(raw_id)
        except (TypeError, ValueError):
            raise ValueError(f"Некорректный symbol_id: {raw_id!r}") from None
        descriptions[symbol_id] = description
    return descriptions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file", type=Path, help="JSON {symbol_id: description}")
    parser.add_argument(
        "--db", type=Path, default=Path("data/bsl_index.db"), help="путь к SQLite-индексу"
    )
    args = parser.parse_args()

    try:
        descriptions = _read_descriptions(args.json_file)
        from src.storage.sqlite_store import SQLiteStore

        result = SQLiteStore(args.db).load_generated_docs(descriptions)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Ошибка импорта: {error}", file=sys.stderr)
        return 2

    print(
        "Импортировано: "
        f"получено={result['received']}, обновлено={result['updated']}, "
        f"некорректно={result['invalid']}, всего_со_стикером={result['total_with_description']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
