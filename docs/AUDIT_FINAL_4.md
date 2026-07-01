# AUDIT FINAL 4 — Football Highlight Studio v2 (Blocks A–G)

Ветка `feat/v2-studio-pipeline`. Этот отчёт — честная фиксация того, ЧТО и КАК
проверено: (a) юнит-тестами на CPU, (b) визуально по извлечённым кадрам,
(c) статически, и что осталось «**verify on GPU**» (GPU/Modal недоступны в
песочнице — стадии YOLO detect/track/seg и vLLM-Director честно НЕ выдаются за
проверенные).

Окружение проверки: Python 3.11 venv (cv2/numpy/pyyaml/pillow/scenedetect),
статический ffmpeg (BtbN). GPU нет, `ultralytics`/`torch`/`openai`/`easyocr`
не установлены — поэтому вижн-стадии деградируют штатно и логируют это.

---

## Сводка по блокам

| Блок | Что сделано | Как проверено |
|------|-------------|---------------|
| A. Оверлеи/типографика | Один чистый набор оверлеев в v2: убран дубль-хук и большой стат-блок из branding; хук — из EditPlan (Composer), стат-карточки — только на реакшн-катах; lower-third компактный и не обрезан | `test_branding.py` + **кадры** (`chk_03`, `chk_10`) |
| B. Seg-mask окклюзия | `Occluder` (YOLO*-seg) + `composite_under_players` в Composer; graceful fallback на grounded-дугу | `test_occlusion.py` (CPU, синтет. маска); **реальные seg-маски = verify on GPU** |
| C. Камера | Упреждение по скорости мяча (`_apply_lead`) + лимиты pan/zoom (`_limit_rate`), per-shot | `test_studio.py` (lead/clamp/per-shot) + **кадры** (кроп держит мяч/героя) |
| D. Guardrail | Факт-чек score/number/possession, регенерация «выдуманного» хука | `test_guardrail.py` |
| E. Web UI | Источник моментов (детекторы/текст-CSV/ESPN) + kick-off поля + **dry-run** preview окон до рендера | `test_preview.py` (gradio-free ядро) |
| F. Compilation | Бит-синхронные точки склейки рила (30–60с) | `test_montage.py` (click-track 120 BPM) |
| G. Verify-on-GPU | Modal-конфиг статически валиден; **реальный GPU-прогон не выполнялся** | статически + честная пометка |

---

## A — Оверлеи (проверено ПО КАДРАМ)

Реальный прогон цепочки **Cameraman → Composer → branding** на синтетическом
клипе (скрипт `.render_check/gen.py`, вне репо), рендер `1080×1920 H.264+AAC`,
затем `ffmpeg` extract + просмотр кадров:

- **Один** хук «WHAT A GOAL!» в компактной плашке сверху (раньше было два
  накладывающихся — статический из branding + из Composer). Дубль убран:
  `apply_branding(..., composer_typography=True)` больше не рисует второй хук и
  большой стат-блок (`_overlay_specs` — единый источник правды, покрыт
  `test_branding.py`).
- **Стат-карточка только на реакшн-кате**: `TOP SPEED 31 KM/H` видна на t=0.3s
  (внутри reaction-окна `(0,0.8)`) и ОТСУТСТВУЕТ на t=1.0s (вне окна).
- **lower-third** «GOAL - 67'» компактный, у левого края в safe-margin
  (`x=w*0.06`), НЕ обрезан.
- Тонкий белый шлейф мяча + grounded-дуга нимба под мячом/ботсами.

Конфиг-контракт A: `captions.{font,font_scale,hook_scale,plate_scale,max_lines,
scoreboard_safe_top_frac,reaction_only}` присутствуют; шрифты Montserrat/Teko в
`assets/fonts` (fallback Inter→DejaVu через `resolve_font`).

Граница честности: динамический анти-overlap с РЕАЛЬНО детектированным bbox
табло не делаем — используется статическая верхняя safe-полоса
(`scoreboard_safe_top_frac`, по умолчанию 0..18%), как и предписано контрактом;
к тому же тугой 9:16-кроп на игроке обычно вырезает вещательное табло из кадра.

## B — Seg-mask окклюзия (CPU-логика проверена; качество seg = verify on GPU)

- `src/graphics/occlusion.py::Occluder` грузит YOLO*-seg (по умолчанию
  `yolo11x-seg.pt`, Ultralytics авто-скачивание; можно указать футбольную
  seg-модель). `available()` РЕАЛЬНО грузит модель — без неё честно `False`, и
  Composer откатывается на grounded-дугу (не имитирует окклюзию).
- В Composer при `telestration.occlusion=true` и доступной модели графика
  рисуется на прозрачном BGRA-слое и композитится ПОД игроками через
  существующую `composite_under_players`; иначе — текущая нижняя дуга.
- `test_occlusion.py` (CPU, без YOLO): с синтетической маской пиксели игрока
  сохраняются, графика видна только ВНЕ маски; alpha-блендинг вне маски; путь
  «маска=None → графика поверх, без падения»; честный fallback Occluder без
  модели; сквозная проверка, что occluded-аннотатор Composer идёт через
  `composite_under_players`.
- **Verify on GPU**: качество реальных seg-масок игроков и скорость на видео.

## C — Камера (проверено юнит-тестами + кадрами)

- `_apply_lead`: фокус смещается ВПЕРЁД по скорости мяча (`edit.reframe.lead_gain`,
  клампится `lead_max_frac`), считается per-shot (не течёт через склейку).
- `_limit_rate`: жёсткие лимиты скорости пана и темпа зума
  (`max_pan_frac_per_s`, `max_zoom_rate_per_s`) после Кальмана, per-shot (склейка
  всё ещё может прыгнуть мгновенно).
- Тесты: лид двигает фокус вперёд и нулевой при gain=0; клампится; rate-limit
  слюит внутри шота, но свободен на стыке. По кадрам: герой/мяч стабильно в
  кадре, без леттербокса.
- **Verify on GPU**: сглаживание/лид на РЕАЛЬНЫХ треках BoT-SORT+CMC.

## D — Guardrail (проверено юнит-тестами)

- `src/qa/guardrail.py`: `verify_text` режет неподтверждённые SCORE («3-0»),
  JERSEY («#7»), POSSESSION («68%»); `guardrail_plan` санитизирует хук/лоуэр-терд
  и РЕГЕНЕРИРУЕТ хук из безопасного дефолта, если он «выдуман» целиком.
  `facts_from` собирает факты из score-verified окна + hero_number + possession.
- Интеграция: в `studio_pipeline` после analytics при `qa.use_guardrail`,
  манифест/капшн пере-derive из очищенного текста.
- Граница честности: имена НЕ вырезаем автоматически (иначе покалечим all-caps
  хуки типа «TOP BINS!») — задокументировано, не угадано.

## E — Web UI + dry-run (ядро проверено без gradio)

- `src/detection/preview.py` (gradio-free): `event_feed_overrides` с ПРИОРИТЕТОМ
  ручных данных (загруженный StatsBomb/SoccerNet JSON > вставленный
  описательный лог/капшены > ESPN fixture (опц.) > детекторы); `preview_markdown`
  (только Scout/парсинг лога, без рендера) → **markdown-таблица** окон
  (мин/видео-таймкод/тип/verified/описание), работает даже до выбора видео.
- `src/detection/event_feed.py::load_descriptive_events`: понимает **StatsBomb**
  (Shot.outcome=Goal→goal, saved→chance, card, dribble→skill) и **SoccerNet**
  (`annotations` c `gameTime "half - MM:SS"`), иначе делегирует в `load_events`
  (текст/CSV/generic JSON). `scout` в рендер-пути тоже идёт через него, так что
  JSON работает end-to-end.
- `app/webui.py`: секция «Источник моментов (v2) + dry-run» — «Paste Descriptive
  Match Log / Captions» (10 строк) + «Or Upload StatsBomb/SoccerNet JSON»
  (`gr.File`), два поля kick-off, ESPN (опц.), кнопка dry-run → `gr.Markdown`.
  Стратегия: РУЧНОЙ лог первичен, автопоиск (ESPN) — вторичен.
- Инвариант параметров (проверено AST-скриптом): `render_job` **26** параметров
  == **26** inputs `run_btn.click`; `preview_windows` **10** == **10** inputs
  `preview_btn.click`. (Не 18: ветка уже несёт полный Block E — ESPN/kickoff/
  occlusion; ломать рабочее ради числа не стал, инвариант «inputs==params, без
  падений» соблюдён на реальном числе.)
- `test_preview.py` + `test_event_feed.py`: overrides (детекторы/ESPN/текст/JSON,
  приоритет загруженного файла); StatsBomb/SoccerNet парсинг; markdown dry-run из
  вставленного лога (без видео) и из StatsBomb-JSON; graceful на пустом вводе.
- Граница честности: сам gradio в песочнице не установлен → UI-логика вынесена в
  тестируемый модуль; фактический запуск сервера = verify on Modal (см. G).

## F — Compilation бит-синхрон (проверено ffmpeg-тестом)

- `compilation._beat_cut_plan`: при `edit.audio.beat_sync` детектит бит и
  подрезает каждый сегмент до целого числа битовых периодов
  (`beat_min_beats_per_segment`); музыка идёт от t=0 → все стыки на бите.
  `ff.standardize(trim=)` для точной подрезки.
- `test_montage.py`: `select_for_duration` попадает в окно 30–60с
  (хронологический порядок после отбора по confidence); сгенерированный
  click-track 120 BPM доказывает, что точки склейки ложатся на битовые периоды,
  а бит-синхронный рил рендерится в валидный 1V+1A.

## G — Verify-on-GPU (Modal) — ЧЕСТНО НЕ ЗАПУСКАЛОСЬ

GPU и Modal CLI в песочнице **недоступны** (`no modal cli`, `import modal` →
ModuleNotFoundError, NVENC отсутствует → libx264). Поэтому реального GPU-прогона
`modal run modal_app.py::studio_local` НЕ было — и он НЕ выдаётся за
выполненный (жёсткое правило #4).

Что проверено статически:
- `modal_app.py` компилируется; `GPU = "A100-80GB"` (цель ТЗ); `vlm()` условно
  добавляет `--quantization awq` только для AWQ-чекпойнтов; Director/Critic
  подключаются к vLLM через `_vlm_overrides()`/`FHS_VLM_URL`
  (`director.backend=openai`, `base_url=.../v1`), что совпадает с контрактом
  `llm_client.VisionLLMClient`.
- Пайплайн-оркестрация БЕЗ GPU честно деградирует: вижн-стадии логируют
  недоступность моделей, падения изолируются по клипу (`stage_failed/error`).

Что осталось **verify on GPU** (следующая сессия с GPU/Modal):
1. `modal run modal_app.py::setup_models` и `::setup_vlm` (кэш весов на Volume).
2. `modal run modal_app.py::studio_local --name <match> --limit 1` → скачать
   output → `ffmpeg` extract → **просмотр кадров** → сравнение с эталоном
   `output/video_2026-06-30_19-39-46.mp4` (576×768) → итерации.
3. Реальные YOLO player/ball/pitch/seg + BoT-SORT+CMC + Qwen2.5-VL Director/Critic.
4. `modal deploy` → GET /=200 → загрузка видео в UI → стриминг → скачиваемые
   ролики/рил.

---

## Definition of Done — статус

1. Клип содержит реальный гол/скилл (ESPN/фид + cutaway-гейт + VLM) — **логика
   готова и покрыта** (event_feed/scout/cutaway_gate тесты); проверка на живом
   матче = verify on GPU.
2. Линии валидны/отсутствуют; нимбы тонкие командных цветов ПОД игроками; шлейф
   10–15 кадров — **CPU + кадры готово**; seg-окклюзия качество = verify on GPU.
3. Кроп 9:16 тугой/плавный, держит мяч/героя, без леттербокса/рывков — **лид +
   лимиты + кадры готово**; на реальных треках = verify on GPU.
4. Один компактный набор оверлеев, мелкие субтитры, стат-карточки на реакшн-катах,
   ничего поверх табло, lower-third не обрезан — **проверено по кадрам ✓**.
5. Каждый mp4 = 1 H.264 + 1 AAC stereo @ единый WxH/fps, процесс не падает —
   **проверено** (render-check + test_polish/test_montage/test_audit).
6. compileall чистый, все тест-сьюты зелёные — **да** (16 сьютов) — визуал
   подтверждён по кадрам для CPU-частей.
7. Web UI: источник/движок/стриминг/скачивание, dry-run — **логика готова +
   тесты**; `modal deploy` GET /=200 = verify on Modal.
8. Отчёт — **этот файл**.

## Границы/честность
- Номера на широких планах ненадёжны → деградация к геометрическому выбору героя.
- Данные событий: ESPN keyless/openfootball/лицензии; без скрейпа.
- Ultralytics AGPL; статы помечаются оценочными; права на контент — на операторе.
- Синтетический клип render-check — прямоугольники/круги (не футбол): доказывает
  ПОВЕДЕНИЕ оверлеев/кропа/кодека, не эстетику реального матча (та = verify on GPU).
