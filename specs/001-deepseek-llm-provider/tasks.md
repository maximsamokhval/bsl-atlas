---
description: "Task list for feature 001-deepseek-llm-provider"
---

# Tasks: DeepSeek LLM Provider Integration

**Input**: Design documents from `/specs/001-deepseek-llm-provider/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: TDD required per constitution – each implementation task must be preceded by a failing test.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Single project structure as defined in plan.md:
- `src/` at repository root
- `tests/` at repository root

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create directory structure and prepare for new modules.

- [x] T001 [P] Create LLM module directories: `src/llm/`, `tests/llm/`
- [x] T002 [P] Add empty `__init__.py` files in `src/llm/` and `tests/llm/`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core abstractions and configuration that MUST be complete before ANY user story can be implemented.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T003 [P] Implement LLMProvider protocol in `src/llm/base.py`
- [x] T004 [P] Extend EmbeddingProvider to support "deepinfra" in `src/indexer/embeddings.py`
- [x] T005 [P] Add DeepSeek and Deepinfra configuration fields to `Config` class in `src/config.py`
- [x] T006 [P] Write failing test for LLMProvider protocol in `tests/llm/test_base.py`
- [x] T007 [P] Write failing test for deepinfra embedding provider in `tests/indexer/test_embeddings_deepinfra.py`

**Checkpoint**: Foundation ready – user story implementation can now begin in parallel.

---

## Phase 3: User Story 1 - Использование DeepSeek API для генерации текста (Priority: P1) 🎯 MVP

**Goal**: Пользователь может отправлять текстовые запросы через DeepSeek API и получать корректные ответы.

**Independent Test**: Запустить тест `tests/llm/test_deepseek.py` с моком HTTP‑запроса – тест должен пройти после реализации.

### Tests for User Story 1 (TDD required)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T008 [P] [US1] Write failing test for DeepSeekProvider in `tests/llm/test_deepseek.py`

### Implementation for User Story 1

- [x] T009 [P] [US1] Implement DeepSeekProvider in `src/llm/deepseek.py`
- [x] T010 [US1] Implement factory function `create_llm_provider` in `src/llm/factory.py`
- [x] T011 [US1] Integrate LLM provider into existing flow (optional) – update `src/main.py` or CLI entry point

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently.

---

## Phase 4: User Story 2 - Конфигурируемый провайдер эмбеддингов (Priority: P2)

**Goal**: Оператор может выбирать провайдера эмбеддингов (Ollama / Deepinfra) через переменные окружения.

**Independent Test**: Запустить тест `tests/indexer/test_embeddings_deepinfra.py` с моком Deepinfra API – тест должен пройти после реализации.

### Tests for User Story 2 (TDD required)

- [x] T012 [P] [US2] Write failing integration test for deepinfra embedding provider in `tests/indexer/test_embeddings_deepinfra.py` (if not covered by T007)

### Implementation for User Story 2

- [x] T013 [P] [US2] Implement deepinfra embedding provider case in `src/indexer/embeddings.py`
- [x] T014 [US2] Update `create_embedding_provider` factory to support "deepinfra" in `src/indexer/embeddings.py`
- [x] T015 [US2] Add configuration validation for deepinfra API key in `src/config.py`

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently.

---

## Phase 5: User Story 3 - Соблюдение ограничений проекта (Priority: P3)

**Goal**: Реализация соответствует трём ключевым ограничениям конституции (Brownfield, TDD, стандарты качества).

**Independent Test**: Все новые файлы проходят линтер, типизацию, тесты и security check; в `git diff` нет изменений оригинального кода.

### Implementation for User Story 3

- [x] T016 [P] [US3] Run linter (`ruff`) on all new files and fix any errors
- [x] T017 [P] [US3] Run type checker (`mypy`) on all new files and fix any errors
- [x] T018 [US3] Run all tests (`pytest`) and ensure they pass (green)
- [x] T019 [P] [US3] Run security check (`bandit`) on all new files and fix any warnings
- [x] T020 [US3] Verify no original repository files have been modified (`git diff` against main branch)

**Checkpoint**: All user stories should now be independently functional and compliant with constitution.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories.

- [x] T021 [P] Update project documentation (README.md) with DeepSeek and Deepinfra setup instructions
- [x] T022 [P] Update quickstart.md with any missing steps
- [x] T023 [P] Add usage examples in `examples/` or `docs/`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies – can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion – BLOCKS all user stories.
- **User Stories (Phase 3+)**: All depend on Foundational phase completion.
  - User stories can then proceed in parallel (if staffed).
  - Or sequentially in priority order (P1 → P2 → P3).
- **Polish (Phase 6)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) – No dependencies on other stories.
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) – May integrate with US1 but should be independently testable.
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) – Depends on completion of US1 and US2 (since it validates their outputs).

### Within Each User Story

- Tests (TDD) MUST be written and FAIL before implementation.
- Models before services, services before integration.
- Story complete before moving to next priority.

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel.
- All Foundational tasks marked [P] can run in parallel (within Phase 2).
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows).
- All tests for a user story marked [P] can run in parallel.
- Models within a story marked [P] can run in parallel.
- Different user stories can be worked on in parallel by different team members.

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Write failing test for DeepSeekProvider in tests/llm/test_deepseek.py"

# Launch implementation tasks after test passes:
Task: "Implement DeepSeekProvider in src/llm/deepseek.py"
Task: "Implement factory function create_llm_provider in src/llm/factory.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational (CRITICAL – blocks all stories).
3. Complete Phase 3: User Story 1.
4. **STOP and VALIDATE**: Test User Story 1 independently.
5. Deploy/demo if ready.

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready.
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!).
3. Add User Story 2 → Test independently → Deploy/Demo.
4. Add User Story 3 → Test independently → Deploy/Demo.
5. Each story adds value without breaking previous stories.

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together.
2. Once Foundational is done:
   - Developer A: User Story 1.
   - Developer B: User Story 2.
   - Developer C: User Story 3.
3. Stories complete and integrate independently.

---

## Notes

- [P] tasks = different files, no dependencies.
- [Story] label maps task to specific user story for traceability.
- Each user story should be independently completable and testable.
- Verify tests fail before implementing.
- Commit after each task or logical group.
- Stop at any checkpoint to validate story independently.
- Avoid: vague tasks, same file conflicts, cross‑story dependencies that break independence.
