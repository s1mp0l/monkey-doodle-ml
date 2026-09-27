# monkey-doodle-ml

Классификатор рисунков для «Monkey School»: игрок рисует мышью на листе или доске, модель на устройстве
(Unity Inference Engine, ONNX) говорит, что нарисовано — одно из заданий урока, `other` («не разобрал») или
`rude` (неприличный рисунок). Обучение — на Google Quick, Draw! (CC BY 4.0, см. [ATTRIBUTION.md](ATTRIBUTION.md))
плюс свои рисунки, собранные в редакторе этого проекта.

```
config/classes.yaml        классы: 31 задание (классы Quick Draw, task: true) + other (60 источников) + rude
config/lesson_prompts.json список заданий для Unity (id, имя Quick Draw, ru, ru_acc) — генерируется
doodle/preprocess.py       КАНОНИЧЕСКАЯ предобработка (спецификация: docs/PREPROCESS.md)
doodle/gpu_raster.py       та же растеризация батчем на GPU + аугментации на штрихах
doodle/model.py            DoodleNet: 4×(2 conv3×3-BN-ReLU) → GAP → FC, ~1.2 M параметров
doodle/data.py, config.py, ndjson_io.py
scripts/                   download, prepare, train, evaluate, export_onnx, editor, import_contrib, lesson_prompts, make_test_vectors
editor/                    редактор датасета (http.server + чистый JS)
webapp/                    публичный сборщик рисунков без бэкенда (статический сайт для Vercel)
tests/                     pytest + tests/vectors/ (эталон предобработки для порта на C#)
```

Все скрипты: `python scripts/<имя>.py --help`. Команды ниже — из корня проекта, Windows (Git Bash:
`.venv/Scripts/python`, PowerShell: `.venv\Scripts\python`).

## Установка

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
# PyTorch с CUDA. Драйвер 610.74 = CUDA 13.3 (nvidia-smi) → колёса cu132:
.venv\Scripts\python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → 2.14.0+cu132 True NVIDIA GeForce RTX 4060 Ti
.venv\Scripts\python -m pytest
```

Другой драйвер: смотрите «CUDA Version» в `nvidia-smi` и берите ближайший не больший индекс
`https://download.pytorch.org/whl/cu1xx`. Экспорт использует классический (TorchScript) экспортёр
`torch.onnx.export(..., dynamo=False)` — только он честно пишет opset 15; новый (dynamo) всегда даёт opset 18.
Если в будущей версии PyTorch его уберут — держите torch 2.14 для экспорта.

## Конвейер

```powershell
# 1. Скачать Quick Draw (simplified ndjson, только recognized, лимит на класс, докачка)
.venv\Scripts\python scripts/download.py --estimate          # размеры без скачивания
.venv\Scripts\python scripts/download.py                     # полный набор: 91 источник, до 710 тыс. рисунков
# 2. Подготовить штрихи + split (train/val/test = 90/5/5 по crc32(key_id), стабильно при добавлении данных)
.venv\Scripts\python scripts/prepare.py                      # → data/prepared/full
# 3. Обучить (GPU)
.venv\Scripts\python scripts/train.py --data data/prepared/full   # → runs/<время>/
# 4. Оценить: accuracy, precision/recall, матрица ошибок, калибровка (T), пороги по классам
.venv\Scripts\python scripts/evaluate.py --run runs/<время>
# 5. Экспорт ONNX + labels.json (+ fp16), проверка паритета и скорости на CPU
.venv\Scripts\python scripts/export_onnx.py --run runs/<время>    # → models/<время>/ и models/latest/
```

Быстрый прогон на подмножестве: `--profile smoke` у `download.py` и `prepare.py`
(8 классов × 5000 + other из 10 классов × 1000), затем `train.py --data data/prepared/smoke --epochs 8`.

| Что | Куда |
|---|---|
| Quick Draw | `data/raw/quickdraw/<класс>.ndjson` (+ `.state.json` для докачки) |
| свои рисунки | `data/custom/<класс>.ndjson`, журнал удалений/пометок `data/custom/_review.ndjson` |
| подготовленные данные | `data/prepared/<профиль>/` (`meta.json`, `*_pts/sid/start/len/label.npy`, `val/test_images.npy`) |
| прогоны | `runs/<время>/` — `best.pt`, `last.pt`, `metrics.csv`, `aug_samples.png`, `eval/` |
| модели | `models/<время>/` и `models/latest/` — `doodle.onnx`, `doodle_fp16.onnx`, `labels.json`, `export_report.json` |

`data/`, `runs/`, `models/` — в `.gitignore`.

### Объёмы и время (RTX 4060 Ti)

- Полный набор (`defaults.cap: 20000` на задание, `other` — 60 × 1500): **≈ 330 МБ** чтения из бакета,
  ≈ 310 МБ на диске, скачивание — минуты; `prepare.py` — ~1–2 мин, ≈ 0.4 ГБ в `data/prepared/full`.
- Обучение: ≈ 4.7 тыс. примеров/с (растеризация и аугментации на GPU, bf16) → **~2.5 мин на эпоху**,
  30 эпох ≈ 75 мин (ранняя остановка обычно раньше). Smoke (45 тыс.) — 10 с на эпоху.
- Больше данных почти бесплатно: `download.py --cap 50000` (≈ 0.8 ГБ) → эпоха ~6 мин. Для 64×64 и ~33
  классов 20–50 тыс. на класс — разумный диапазон; Quick Draw даёт 100–300 тыс. на класс.
- `rude`: цель **≥ 3000** рисунков от разных людей (минимум для запуска — `min_samples: 500`, иначе громкое
  предупреждение). Пока `rude` пуст, он просто исключается из модели.

## Предобработка — одна на всё

strokes → нормализация «как Quick Draw simplified» (сдвиг к (0,0), масштаб до 255, RDP ε = 2, округление) →
растр N×N «расстояние до ближайшего отрезка» с фиксированной толщиной, полями и центрированием.
Полная спецификация для порта на C#, включая то, как подать точки `DrawingSurface` (Y вверх, пиксели
текстуры), — [docs/PREPROCESS.md](docs/PREPROCESS.md). Эталонные векторы — `tests/vectors/*.json|npy|png`
(перегенерация: `scripts/make_test_vectors.py`, только при осознанной смене спецификации).

Аугментации при обучении (на штрихах, до растеризации): поворот, анизотропный масштаб, сдвиг, зеркало,
дрожание точек, случайное уменьшение/сдвиг в кадре, толщина линии, выкидывание штрихов у длинных рисунков.
Класс с `augment: heavy` (`rude`) получает всё сильнее. Сбалансированная выборка: доля класса
`sampling_weight · min(1, n/median)^0.5` — маленький `rude` повторяется, но не тысячи раз за эпоху.

## Редактор датасета

```powershell
.venv\Scripts\python scripts/editor.py            # http://127.0.0.1:8765
.venv\Scripts\python scripts/editor.py --lan      # 0.0.0.0:8765 — рисовать с других ПК/телефонов в сети
```

- **Рисовать**: мышь, перо и палец (Pointer Events), вид мелка, цвета, формат «квадрат / лист / доска»;
  отменить (Ctrl+Z), очистить (Esc), пропустить, сохранить (Enter). Класс (по умолчанию `rude`), имя автора
  (запоминается), счётчик и цель, подсказки разнообразия по одной («маленький», «повёрнутый», «одной линией» …).
- Сохраняется дописыванием в `data/custom/<класс>.ndjson`: `key_id, word, author, timestamp, prompt, device,
  canvas, drawing` (Quick Draw raw `[[xs],[ys],[ts]]`) и `simplified`. Запись — одним `write` под
  межпроцессной блокировкой + `fsync`; оборванная строка не ломает файл; несколько клиентов одновременно — ок.
- **Галерея**: миниатюры через каноническую предобработку (ровно вход модели), фильтры класс/автор/статус,
  «пометить» / «удалить» / «вернуть» (мягко, журнал `_review.ndjson`; `prepare.py` пропускает удалённые и
  помеченные).
- **Тест**: рисуете — top-5 текущей `models/latest/doodle.onnx` на лету (onnxruntime на сервере), вердикт
  с порогом класса и картинка «так видит модель».
- Авторизации нет — `--lan` только в доверенной сети.

## Веб-сборщик рисунков (`webapp/`)

Статический сайт без сервера: участники рисуют `rude` в браузере и присылают вам файл. Та же рисовалка, что во
вкладке «Рисовать» редактора (мышь/перо/палец, мелок, отмена, Enter, подсказки разнообразия), карточка
согласия, необязательный ник. Рисунки хранятся в IndexedDB браузера (переживают перезагрузку), есть счётчик
и галерея своих рисунков с удалением. «Скачать мои рисунки» отдаёт один `.ndjson`
(`rude_<ник>_<дата>_<N>.ndjson`) в формате записей редактора (`key_id` — UUID, `source: "web"`,
`simplified` считает JS-порт канонического упрощения — совпадает с `doodle/preprocess.normalize` бит-в-бит,
проверяется в pytest через node). После отправки — «Очистить отправленные».
Ни сетевых запросов, ни cookie, ни аналитики; CSP `default-src 'none'`, `connect-src 'none'`.

- Классы — константа `CONFIG.classes` в `webapp/core.js` (сейчас только `rude`; добавьте класс — появится
  выбор). Там же `CONFIG.sendHint` — куда слать файл.
- Локально: `.venv\Scripts\python -m http.server 8080 --bind 127.0.0.1 --directory webapp` → http://127.0.0.1:8080
  (заголовки из `vercel.json` локально не применяются).

### Деплой на Vercel

```powershell
cd webapp
npx vercel login          # один раз
npx vercel                # первый раз: создать проект (framework: Other, build — нет, output — .)
npx vercel --prod         # публикация
```

`webapp/vercel.json` — статика без сборки, заголовки безопасности (CSP, `X-Frame-Options: DENY`,
`Referrer-Policy: no-referrer`, `Permissions-Policy`, `X-Robots-Tag: noindex`).

### Импорт присланных файлов

```powershell
.venv\Scripts\python scripts/import_contrib.py inbox\            # папка или файлы .ndjson
.venv\Scripts\python scripts/import_contrib.py inbox\ --dry-run  # только проверить
```

Проверяет каждую запись (схема, 1–200 штрихов, 2–20000 точек, границы координат и холста, время не убывает,
≤ 10 мин, не точка), отбрасывает дубли по `key_id` и по хешу координат (в пачке и против уже импортированного),
ставит `source: "web"`, `contributor`, `imported_at`, пересчитывает `simplified` эталоном и одной атомарной
дозаписью кладёт в `data/custom/rude.ndjson`. Печатает по каждому файлу: принято / отклонено с причинами.
Дальше — просмотр в галерее локального редактора (пометить/удалить плохие).

## В Unity

1. Пакет **Inference Engine** (`com.unity.ai.inference`, бывший Sentis) через Package Manager. Он
   поддерживает ONNX opset 7–15 — экспорт делается в opset 15; в графе только Conv, Relu, MaxPool,
   GlobalAveragePool, Flatten, Gemm, Softmax (BatchNorm и температура калибровки влиты в веса).
2. Скопировать `models/latest/doodle.onnx` (или `doodle_fp16.onnx` — вдвое меньше, веса fp16, расчёт fp32;
   либо fp32 + квантизация Float16 в самом Unity) и `labels.json` в проект, например
   `Assets/_Project/Domains/Lesson/Data/Doodle/`. Туда же — `config/lesson_prompts.json` (список заданий:
   `id` = метка модели, `ru` — подпись, `ru_acc` — для «Нарисуйте: …»).
3. Портировать предобработку по `docs/PREPROCESS.md` и прогнать `tests/vectors/` как EditMode-тесты.
4. Инференс (набросок):

```csharp
using Unity.InferenceEngine;
var model  = ModelLoader.Load(modelAsset);                    // ModelAsset из doodle.onnx
var worker = new Worker(model, BackendType.CPU);              // 64×64 — CPU хватает (~2.5 мс на ПК, 1 поток)
using var input = new Tensor<float>(new TensorShape(1, 1, 64, 64), pixels);  // pixels[row*64+col], 1 = чернила
worker.Schedule(input);
using var probs = (worker.PeekOutput() as Tensor<float>).ReadbackAndClone();
// top-1; если id == "other" или p < threshold класса из labels.json → «не разобрал»
```

Распознавать разумно по окончании штриха/по кнопке «Сдать», не каждый кадр.

## Тесты

`pytest` — векторы предобработки, совпадение GPU-растеризатора с эталоном, ndjson (конкурентная запись,
оборванные строки, журнал ревью), редактор через HTTP (сохранение, галерея, ревью, предсказание,
6 параллельных клиентов), импорт веб-файлов (валидные, битые, дубли), паритет JS-упрощения с эталоном,
синхронность `lesson_prompts.json` с конфигом.
