# ТЗ: доведение Football Highlight Studio (v2) до эталонного качества

> Цель — на выходе ролик в стиле эталона `output/video_2026-06-30_19-39-46.mp4`
> (быстрый скилл-монтаж 9:16, тонкие неоновые HUD-элементы, чистая типографика),
> ИЛИ как минимум аккуратная нарезка реальных голов/скиллов с неоновыми halo/
> шлейфом, которые **не улетают в космос**.
> Документ привязан к реальному коду ветки `feat/v2-studio-pipeline`.

---

## 0. Definition of Done

1. **Моменты — настоящие.** Клип содержит гол/скилл (подтверждён данными матча
   и/или VLM), без «20 секунд скамейки». Отбор идёт от событийного фида, а не от
   OCR-угадывания.
2. **HUD не «улетает».** Тактические линии либо корректны, либо отсутствуют;
   нимбы тонкие (≤2px), командных неоновых цветов, ПОД игроками; шлейф мяча 10–15
   кадров, белый, без «лучей в небо».
3. **Кроп 9:16 тугой и плавный** — держит мяч/героя, без статичного леттербокса и
   без дёрганья.
4. **Один компактный набор оверлеев**, типографика Montserrat/Teko, мелкие
   субтитры, ничего поверх лиц/мяча/вещательного табло.
5. **Каждый mp4** = 1 H.264 video + 1 AAC stereo @ единый WxH/fps.
6. **Процесс не падает**: изоляция падений по клипу, видимые ошибки, graceful
   degradation без GPU/моделей/данных.
7. **Тесты зелёные**, компиляция чистая, верификация по КАДРАМ (render→extract→
   view→compare), а не только по unit-тестам.

---

## 1. Целевая архитектура и поток данных

```
                 ┌─────────────────────── ИСТОЧНИКИ МОМЕНТОВ ───────────────────────┐
 match video ──► │  A. event_feed (ESPN/CSV/текст) ──► окна по kick-off офсету        │
                 │  B. scoreboard OCR  ──► коррекция дрейфа окон голов (align_to_ocr) │
                 │  C. audio peaks / action-spotting ──► fallback, если фида нет      │
                 └───────────────────────────────┬──────────────────────────────────┘
                                                  ▼  EventWindow[]  (kind,t,conf,verified,meta)
                                          extract_clips()  ──► короткие клипы
                                                  ▼ (на каждый клип, изолированно)
   ┌──────────────────────────── PER-CLIP STUDIO CHAIN ────────────────────────────┐
   │ 1) shots        segment_shots + mark_duplicate (PySceneDetect)                 │
   │ 2) track        Cameraman.track_only  (BoT-SORT + CMC), 1 раз                  │
   │ 3) DIRECTOR     plan_edit(bundle)  ── VLM смотрит кадры ──► EditPlan            │
   │                  • keep_clip (drop скамейки/реплея)                            │
   │                  • per-shot keep/zoom, slow-mo beats, hook-текст               │
   │ 4) homography   compute_homography (опц.) + ВАЛИДАЦИЯ (skip, если плохо)       │
   │ 5) analytics    teams(цвета) + jerseys(номер) + possession + выбор героя       │
   │ 6) reid         кросс-катовый герой (follow across cuts)                       │
   │ 7) RENDER       графика в ОРИГ. пространстве → crop 9:16 → slow-mo → текст     │
   │ 8) QA+CRITIC    qa_report + critic(кадры) + guardrail(факты) → ≤1 ревизия      │
   │ 9) branding     intro/outro/lower-third                                        │
   └───────────────────────────────────────────────────────────────────────────────┘
                                                  ▼
                       per_clip (N роликов)  |  compilation (1 рил 30–60с, beat-sync)
```

### Координатные пространства (критично для «не улетает в космос»)
- **Графика (halo/шлейф/линии)** рисуется в ПИКСЕЛЯХ ОРИГИНАЛЬНОГО клипа (до crop),
  через `composer.make_annotators(... annotate_world)`.
- **Crop 9:16** применяется ПОСЛЕ графики (`cameraman.render`).
- **Slow-mo + типографика** — полнокадровые, ПОСЛЕ crop (`composer.finish`).
- Любая графика, которая считается в неверном пространстве или по «телепортнувшимся»
  координатам (мяч/линии), даёт артефакт «в небо» → обязательны гейты валидации.

---

## 2. Что добавить / изменить (по модулям)

### 2.1 Отбор моментов (приоритет №1)
| Модуль | Действие | Статус |
|---|---|---|
| `src/detection/event_feed.py` | парсер CSV/JSON/текст + kick-off маппинг + OCR-align | ✅ сделано |
| `src/detection/event_feed.py` | **+ ESPN-источник** `load_from_espn(fixture_id/slug)` (keyless, голы+минуты+карточки), провайдер-агностик dataclass | ⬜ добавить |
| `src/detection/event_feed.py` | **+ xT-веса** важности событий (ранжирование top-N) | ⬜ добавить |
| `src/detection/scout.py` | фид как приоритетный источник (есть), оставить OCR/audio как fallback/коррекцию | ✅ сделано |
| `config/config.yaml` | `detect.event_feed.*` (есть) + `detect.event_feed.espn.*`, `top_n` | 🟡 расширить |

### 2.2 Director / агенты (понимание кадра + надёжность)
| Модуль | Действие |
|---|---|
| `config/config.yaml` | `director.backend` по умолчанию = **VLM** при наличии endpoint (сейчас heuristic → ничего не дропает) |
| `src/agents/director_agent.py` | гейт keep/drop по кадрам + явный детектор cutaway/bench/replay |
| `src/agents/critic.py` | **+ guardrail-факт-чек** (паттерн из SynapseQuill): не показывать стат/хук, не подтверждённый данными; перегенерация при «выдуманном» счёте |
| `src/agents/editplan.py` | per-shot `keep` (триминг скамейки внутри окна), beats slow-mo |

### 2.3 Визуал — HUD «не в космос» (приоритет №2)
| Модуль | Действие | Статус |
|---|---|---|
| `src/render/composer.py` | шлейф мяча: jump-rejection + длина 10–15 + тонкий белый | ✅ сделано |
| `src/render/composer.py` | нимбы тонкие ≤2px, неоновые командные цвета | ✅ сделано |
| `src/render/composer.py` | **окклюзия**: вызвать `composite_under_players` (YOLO-seg маски) — графика ПОД игроками | ⬜ добавить |
| `src/render/composer.py` | типографика Montserrat/Teko (TextClip + Pillow-fallback), мелкие субтитры, safe-zone, анти-overlap с табло | 🟡 частично |
| `src/render/composer.py` | стат-карточки только в окно реакшн-кат | ⬜ добавить |
| `src/graphics/homography.py` | ВАЛИДАЦИЯ H (reprojection error/inliers/границы поля) → skip линий при плохом фите | ⬜ добавить |
| `src/tracking/cameraman.py` | action-centric punch-in (`_auto_zoom`, target_subject_height) | ✅ сделано |
| `src/tracking/cameraman.py` | усилить упреждение по скорости мяча + лимит pan/zoom (без дёрганья) | 🟡 проверить |
| `src/vision/teams.py` | неоновая палитра по команде (team0=красный, team1=жёлтый) | ✅ сделано |

### 2.4 UI / запуск
| Модуль | Действие |
|---|---|
| `app/webui.py` | поля: engine(v1/v2), **event-feed (вставка текста/CSV или ESPN fixture)**, 2 поля kick-off, director backend |
| `app/webui.py` | **dry-run preview** окон (из ClipMaker): показать список выбранных моментов ДО рендера |
| `modal_app.py` | (Modal на паузе) — варианты запуска см. §6 |

---

## 3. Синхронизация процессов (как стадии связаны)

1. **Источник → окна.** `scout_events()` возвращает `EventWindow[]`. Контракт окна
   фиксирован (kind/anchor_t/start/end/confidence/verified/sources/meta).
   Все источники (feed/OCR/audio) приводятся к этому контракту, дальше пайплайн
   их не различает. Это точка единой синхронизации «что резать».
2. **Окна → клипы.** `extract_clips()` режет по `start/end`. Дальше каждый клип
   обрабатывается ИЗОЛИРОВАННО (падение одного не валит батч — `_process` ловит
   исключение, пишет `stage_failed`).
3. **track → director → analytics → render — строгий порядок зависимостей:**
   - `track_only` даёт `frames+meta` (нужно всем ниже);
   - `plan_edit` (Director) читает `bundle`(кадры) + `track` → решает keep/zoom/beats;
   - `analytics` читает `track`(+гомография для метров) → герой/цвета/possession;
   - `render_plan` объединяет: crop-план (cameraman) + аннотаторы (composer) +
     finish(slow-mo/текст). Director-план и analytics ОБА должны быть готовы до render.
4. **QA-петля** оборачивает `render_plan`: рендер → `qa_report` (+critic по кадрам)
   → при провале перепланирование (≤1 ревизия) → повторный рендер. Это замкнутая
   обратная связь «сделал → проверил → исправил».
5. **Прогресс** прокидывается единым `on_progress(stage,pct,msg)` из `run_studio`
   в UI/детачнутую джобу (`_status.json`) — один формат на v2.
6. **Координатная синхронизация** (см. §1): графика ДО crop, текст ПОСЛЕ crop.
   Нарушение порядка = артефакты. Это инвариант, который нельзя ломать.

---

## 4. Поэтапный план (приоритеты)

### P0 — «моменты настоящие» (без этого всё бессмысленно)
- [ ] `event_feed.load_from_espn()` (keyless) + dataclass-маппинг в `MatchEvent`.
- [ ] Включить VLM-Director по умолчанию при наличии endpoint; keep/drop + cutaway-детектор.
- [ ] xT-веса + `top_n` отбор.
- **Готово, когда:** на тест-видео выбираются голы/скиллы, в логе видно keep/drop с причиной; скамейка не попадает.

### P1 — «HUD не улетает»
- [ ] Гомография: гейт валидации → skip линий при низкой уверенности.
- [ ] Окклюзия `composite_under_players` (графика под игроками).
- [x] Шлейф/нимбы (тонкие, неон, jump-reject) — уже в ветке.
- **Готово, когда:** на кадрах нет линий «в небо», нимбы тонкие командных цветов под ногами, шлейф короткий белый.

### P2 — «кроп и типографика как у эталона»
- [x] Action-centric punch-in (есть) → [ ] проверить упреждение/лимиты по кадрам.
- [ ] Montserrat/Teko, мелкие субтитры, safe-zone, анти-overlap табло.
- [ ] Стат-карточки только на реакшн-катах.
- **Готово, когда:** серия кадров показывает тугой плавный кроп и один компактный набор оверлеев.

### P3 — «надёжность и UX»
- [ ] Critic + guardrail факт-чек.
- [ ] UI: event-feed + kick-off поля + dry-run preview окон.
- [ ] compilation-рил с beat-sync.

---

## 5. Конфиг-контракт (новые/ключевые ключи)
```
detect.event_feed: {enabled, source, kickoffs:{1,2,..}, min_importance,
                    align_to_ocr, ocr_align_radius_seconds,
                    espn:{enabled, slug, fixture_id}, top_n}
director: {backend(=vllm при endpoint), confidence_threshold, allow_curation}
telestration: {team_palette, line_thickness(<=2), trail_length(10..15),
              trail_max_jump_frac, possession_plate}
graphics.homography: {min_confidence, max_reproj_error_px, require_in_pitch}
edit.reframe: {mode, target_subject_height, min_zoom, max_zoom, lead_gain}
captions: {font, font_scale, max_lines}
qa: {enabled, use_critic, use_guardrail, max_revisions}
```
**Контракт-чек:** все ключи, что читает код, обязаны присутствовать в `config.yaml`.

---

## 6. Верификация и запуск
- **CPU (без GPU):** `compileall`; unit-тесты (studio/agents/qa/shots/event_feed/
  polish/montage); контракт-чек конфига; импорт `app.webui` + GET / = 200.
- **Обязательно по кадрам:** render→`ffmpeg` extract→просмотр→сравнение с эталоном.
  Без визуальной проверки задача не закрывается.
- **GPU-стадии** (VLM, YOLO-seg, трекинг+CMC, OCR номеров, гомография) — пометить
  «verify on GPU».
- **Запуск (Modal на паузе):** варианты — (a) локально на машине с GPU
  (`run_studio` напрямую), (b) вернуть Modal позже (деплой долгий из-за vllm-образа),
  (c) headless `studio_local`. Решить отдельно.

---

## 7. Лицензии / комплаенс
- Код ClipMaker/SynapseQuill **не копируем** — реализуем идеи заново (проверить их
  лицензии перед любым заимствованием).
- Данные событий: предпочтительно **ESPN keyless / openfootball / лицензированные
  API**; скрейп WhoScored/Scoresway — серая зона ToS, не использовать по умолчанию.
- Ultralytics — AGPL; права на видео-контент матчей — ответственность оператора.
- Статы/хуки помечать честно как оценочные; распознавание номеров на широких планах
  деградирует к геометрическому выбору героя.
```
