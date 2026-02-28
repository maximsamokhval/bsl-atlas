"""Tests for the vector indexing pipeline on a small sample.

Covers three layers:
1. module_type extraction from file paths (pure logic, no deps)
2. BSL parsing — tree-sitter when available, regex fallback otherwise
3. ChromaDB chunk metadata — fake embeddings, in-memory ChromaDB

No real embedding calls (no ollama/openrouter). Runs in seconds.
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers.code import CodeParser
from src.parsers.tree_sitter_parser import is_available as ts_available

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BSL_COMMON_MODULE = """\
Функция НайтиКонтрагента(ИНН) Экспорт
\tЗапрос = Новый Запрос;
\tЗапрос.Текст = "ВЫБРАТЬ Ссылка ИЗ Справочник.Контрагенты ГДЕ ИНН = &ИНН";
\tЗапрос.УстановитьПараметр("ИНН", ИНН);
\tРезультат = Запрос.Выполнить();
\tВозврат Результат.Выгрузить();
КонецФункции

Процедура ОчиститьКэш() Экспорт
\tКэш = Новый Соответствие;
\tОповеститьОбИзменении();
КонецПроцедуры
"""

BSL_OBJECT_MODULE = """\
Процедура ОбработкаПроведения(Отказ, РежимПроведения) Экспорт
\tСтавка = РассчитатьСтавку(Дата);
\tЕсли НЕ СформироватьДвижения(Ставка) Тогда
\t\tОтказ = Истина;
\tКонецЕсли;
КонецПроцедуры

Функция РассчитатьСтавку(Дата) Экспорт
\tВозврат 0.2;
КонецФункции

Процедура СформироватьДвижения_Вспомогательная(Ставка)
\tДвижения.Взаиморасчеты.Записать();
\tДвижения.НДС.Записать();
КонецПроцедуры
"""

# ---------------------------------------------------------------------------
# 1. module_type extraction
# ---------------------------------------------------------------------------

MODULE_TYPE_CASES = [
    ("cf/CommonModules/ОбщегоНазначения/Ext/Module.bsl",    "CommonModule"),
    ("cf/Catalogs/Контрагенты/Ext/ObjectModule.bsl",         "ObjectModule"),
    ("cf/Documents/ЗаказКлиента/Ext/ObjectModule.bsl",       "ObjectModule"),
    ("cf/Catalogs/Контрагенты/Ext/ManagerModule.bsl",        "ManagerModule"),
    ("cf/Documents/ЗаказКлиента/Ext/ManagerModule.bsl",      "ManagerModule"),
    ("cf/AccumulationRegisters/Продажи/Ext/RecordSetModule.bsl", "RecordSetModule"),
    ("cf/Reports/ОтчетПоПродажам/Ext/ObjectModule.bsl",      "ObjectModule"),
    ("cf/DataProcessors/Загрузка/Commands/Загрузить/Ext/CommandModule.bsl", "CommandModule"),
    (
        "cf/Catalogs/Контрагенты/Forms/ФормаЭлемента/Ext/Form/Module.bsl",
        "FormModule",
    ),
]

@pytest.mark.parametrize("rel_path,expected", MODULE_TYPE_CASES)
def test_module_type_extraction(rel_path, expected):
    result = CodeParser._extract_module_type(Path(rel_path))
    assert result == expected, f"path={rel_path}: got {result!r}, want {expected!r}"


# ---------------------------------------------------------------------------
# 2. BSL parsing (tree-sitter or regex)
# ---------------------------------------------------------------------------

def _write_bsl(tmp_path: Path, rel: str, content: str) -> Path:
    """Write BSL content to a file at the given relative path."""
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content.encode("utf-8"))
    return p


def test_parse_common_module(tmp_path):
    path = _write_bsl(
        tmp_path,
        "cf/CommonModules/РаботаСКонтрагентами/Ext/Module.bsl",
        BSL_COMMON_MODULE,
    )
    parser = CodeParser()
    funcs = parser.parse_file_functions(path)

    names = {f.name for f in funcs}
    assert "НайтиКонтрагента" in names
    assert "ОчиститьКэш" in names

    fn = next(f for f in funcs if f.name == "НайтиКонтрагента")
    assert fn.params == ["ИНН"]
    assert fn.is_export is True
    assert fn.module_type == "CommonModule"
    assert fn.type == "Функция"

    proc = next(f for f in funcs if f.name == "ОчиститьКэш")
    assert proc.is_export is True
    assert proc.type == "Процедура"


def test_parse_object_module(tmp_path):
    path = _write_bsl(
        tmp_path,
        "cf/Documents/ЗаказКлиента/Ext/ObjectModule.bsl",
        BSL_OBJECT_MODULE,
    )
    parser = CodeParser()
    funcs = parser.parse_file_functions(path)

    names = {f.name for f in funcs}
    assert "ОбработкаПроведения" in names
    assert "РассчитатьСтавку" in names

    for f in funcs:
        assert f.module_type == "ObjectModule"


def test_tree_sitter_availability():
    """Just report whether tree-sitter is active — not a hard failure."""
    status = "ACTIVE" if ts_available() else "FALLBACK (regex)"
    print(f"\ntree-sitter-bsl: {status}")


def test_parse_calls_extracted(tmp_path):
    """Verify calls are extracted (only meaningful for tree-sitter)."""
    if not ts_available():
        pytest.skip("tree-sitter not available — call extraction may be limited")

    path = _write_bsl(
        tmp_path,
        "cf/CommonModules/Тест/Ext/Module.bsl",
        BSL_COMMON_MODULE,
    )
    parser = CodeParser()
    funcs = parser.parse_file_functions(path)

    fn = next(f for f in funcs if f.name == "НайтиКонтрагента")
    # should detect Запрос.Выполнить or similar calls
    assert len(fn.calls) > 0, f"Expected calls in НайтиКонтрагента, got none"


# ---------------------------------------------------------------------------
# 3. ChromaDB chunk metadata (fake embeddings, in-memory)
# ---------------------------------------------------------------------------

def _fake_embed(texts):
    """Return deterministic 16-dim unit vectors (no network calls)."""
    import hashlib
    result = []
    for t in texts:
        h = hashlib.md5(t.encode()).digest()
        vec = [((b / 255.0) - 0.5) * 2 for b in h]  # 16 floats in [-1,1]
        result.append(vec)
    return result


def test_chromadb_module_type_in_metadata(tmp_path):
    """Index two BSL files → verify module_type stored in ChromaDB metadata."""
    import chromadb
    from chromadb.config import Settings
    from src.indexer.embeddings import ChromaDBEmbeddingFunction, EmbeddingProvider

    # Fake provider
    class FakeProvider:
        def embed_documents(self, texts): return _fake_embed(texts)
        def embed_query(self, text): return _fake_embed([text])[0]

    chroma = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    ef = ChromaDBEmbeddingFunction(FakeProvider())
    col = chroma.create_collection("code", embedding_function=ef)

    # Write two BSL files with different module paths
    common_path = _write_bsl(
        tmp_path,
        "cf/CommonModules/МодульА/Ext/Module.bsl",
        BSL_COMMON_MODULE,
    )
    obj_path = _write_bsl(
        tmp_path,
        "cf/Documents/Документ1/Ext/ObjectModule.bsl",
        BSL_OBJECT_MODULE,
    )

    # Parse + insert chunks directly (replicates what VectorIndexer does)
    parser = CodeParser()
    ids, docs, metas = [], [], []

    for file_path in [common_path, obj_path]:
        funcs = parser.parse_file_functions(file_path)
        for i, fn in enumerate(funcs):
            if fn.body.count("\n") < 3:
                continue
            doc_id = f"{file_path.name}::{fn.name}::{i}"
            ids.append(doc_id)
            docs.append(fn.body[:500])
            metas.append({
                "module_type": fn.module_type,
                "source_file": str(file_path),
                "name": fn.name,
                "is_export": fn.is_export,
            })

    assert len(ids) > 0, "No chunks collected from BSL files"
    col.add(ids=ids, documents=docs, metadatas=metas)

    # Verify all chunks have non-empty module_type
    results = col.get(include=["metadatas"])
    for meta in results["metadatas"]:
        assert meta["module_type"] != "", \
            f"Empty module_type for {meta.get('name')}"
        assert meta["module_type"] in {
            "CommonModule", "ObjectModule", "ManagerModule",
            "FormModule", "RecordSetModule", "CommandModule", "Module",
        }, f"Unknown module_type: {meta['module_type']}"

    # Verify filter by module_type works
    common_results = col.get(where={"module_type": "CommonModule"}, include=["metadatas"])
    obj_results = col.get(where={"module_type": "ObjectModule"}, include=["metadatas"])

    assert len(common_results["ids"]) > 0, "No CommonModule chunks found"
    assert len(obj_results["ids"]) > 0, "No ObjectModule chunks found"

    for meta in common_results["metadatas"]:
        assert meta["module_type"] == "CommonModule"
    for meta in obj_results["metadatas"]:
        assert meta["module_type"] == "ObjectModule"

    print(f"\n  Chunks: {len(ids)} total, "
          f"{len(common_results['ids'])} CommonModule, "
          f"{len(obj_results['ids'])} ObjectModule")
