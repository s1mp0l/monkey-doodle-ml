# Каноническая предобработка рисунка (спецификация для порта на C#)

Одна и та же предобработка используется при обучении (`scripts/prepare.py`, GPU-копия в
`doodle/gpu_raster.py`), в редакторе (`editor/`) и должна быть повторена в Unity.
Эталон — `doodle/preprocess.py`, эталонные векторы — `tests/vectors/`.
Параметры модели лежат в `labels.json → preprocess` — берите их оттуда, а не из кода.

| параметр | значение | смысл |
|---|---|---|
| `epsilon` | 2.0 | допуск Ramer–Douglas–Peucker, в единицах сетки 0..255 |
| `scale_max` | 255 | после нормализации большая сторона рамки = 255 |
| `size` (N) | 64 | сторона входного изображения |
| `pad` | 2.0 | поля, в пикселях выхода |
| `line_width` | 2.0 | толщина линии, в пикселях выхода |

## 0. Вход: штрихи

Рисунок — список штрихов, штрих — список точек `(x, y)` в любых единицах, **ось Y вниз** (как экран и
Quick Draw). Время (`ts`) не используется. Порядок штрихов и точек не важен для результата, кроме RDP
внутри штриха.

### Как подать штрихи из игры

`DrawingSurface` хранит точки штриха (`Stroke.points`, `List<Vector2>`) в **пикселях Texture2D**, ось
**Y вверх** (y = 0 — низ текстуры). Нужно:

1. Брать именно сохранённые точки мыши (`points`), а не пиксели текстуры и не точки сплайна-сглаживания.
   Сейчас список приватный — понадобится геттер `IReadOnlyList<IReadOnlyList<Vector2>> Strokes`.
2. Перевернуть Y: `x' = x`, `y' = Height − y` (любая константа вместо `Height` даёт тот же результат —
   сдвиг убирается нормализацией, важен только знак).
3. Масштаб, размер листа (480×706 лист, 1024×418 доска), радиус и цвет кисти **не важны** — шаг 1
   приводит всё к сетке 0..255. Отмена штриха = просто не передавать удалённый штрих.
4. Точки ближе 1 px к предыдущей игра уже отбрасывает — это совпадает с редактором.

Проверка: вектор `tests/vectors/game_texture_banana.json` содержит `texture.strokes` (координаты игры,
Y вверх, лист 480×706) и ожидаемый результат.

## 1. Нормализация (`normalize`)

1. Выбросить пустые штрихи и точки с нечисловыми координатами. Если штрихов нет — результат пуст,
   изображение из нулей.
2. `minX, minY, maxX, maxY` — по **всем** точкам всех штрихов. `ext = max(maxX − minX, maxY − minY)`.
3. `s = 255 / ext`, если `ext > 0`, иначе `s = 1`.
4. Для каждого штриха по отдельности:
   1. `q = ((x − minX)·s, (y − minY)·s)` для каждой точки;
   2. удалить точки, **в точности** равные предыдущей;
   3. RDP с `epsilon` (ниже);
   4. округлить каждую координату: `floor(v + 0.5)` (не banker's rounding: `Math.Floor(v + 0.5)`);
   5. снова удалить точки, равные предыдущей.
   Штрих из одной точки допустим (тычок).

Результат — целые 0..255, ровно как в Quick Draw *simplified*. Шаг Google «resample 1 px» не делается:
расстояние до хорды выпукло вдоль каждого отрезка, поэтому максимум RDP всегда в исходной вершине
и ресемплинг не меняет результат (проверено тестом `test_resample_is_noop_for_rdp`).

### RDP (итеративно)

```
keep[0] = keep[n−1] = true (n ≥ 3; при n < 3 штрих не меняется)
stack = [(0, n−1)]
while stack не пуст:
    (i, j) = pop
    if j ≤ i + 1: continue
    для k = i+1 .. j−1: d_k = dist_to_segment(p_k, p_i, p_j)
    m = первый k с максимальным d_k        // строго первый при равенстве
    if d_m > epsilon:                     // строго больше
        keep[m] = true; push (i, m); push (m, j)
результат = точки с keep = true в исходном порядке
```

`dist_to_segment(p, a, b)`: `ab = b − a`, `L2 = ab·ab`; если `L2 == 0` → `|p − a|`; иначе
`t = clamp(((p − a)·ab) / L2, 0, 1)`, расстояние `|p − (a + t·ab)|`. Это расстояние до **отрезка**,
не до прямой (важно для замкнутых штрихов, где первая точка = последней).

## 2. Растеризация (`rasterize`)

1. Центр рамки по всем (уже нормализованным) точкам: `cx = (minX + maxX)/2`, `cy = (minY + maxY)/2`.
2. Масштаб фиксированный: `f = (N − 2·pad) / 255` (не по фактической рамке!).
3. Точка → пиксели выхода: `X = N/2 + (x − cx)·f`, `Y = N/2 + (y − cy)·f`.
4. Отрезки: штрих из `n ≥ 2` точек даёт `n − 1` отрезков `(P_k, P_k+1)`; из одной точки —
   вырожденный отрезок `(P, P)`.
5. Для каждого пикселя (строка `row`, столбец `col`, строка 0 — **верх**) центр
   `(col + 0.5, row + 0.5)`; `d` = минимум `dist_to_segment` по всем отрезкам.
6. Интенсивность `I = clamp(line_width/2 + 0.5 − d, 0, 1)` (1 = чернила, 0 = фон).

Вход модели: `float32 [1, 1, N, N]`, `input[0, 0, row, col] = I`, без нормализации среднего/дисперсии.
Unity Inference Engine: `new Tensor<float>(new TensorShape(1, 1, N, N), data)`, где
`data[row * N + col] = I` (row-major, строка 0 — верх изображения).

Для скорости в C# достаточно для каждого отрезка обходить только пиксели его рамки, расширенной на
`line_width/2 + 1` — дальше вклад нулевой. Результат тот же (минимум по отрезкам). 64×64 и ~50 отрезков
— доли миллисекунды.

## 3. Выход модели и решение «узнал / не разобрал»

`probs [1, C]` — уже softmax (температура калибровки влита в последний слой). `labels.json → classes[i]`
описывает индекс `i`. Правило:

```
top = argmax(probs)
if classes[top].id == "other" или probs[top] < classes[top].threshold → «не разобрал»
иначе → узнано classes[top] (для задания урока: верно, если classes[top].id == id задания)
```

`rude` — обычный класс со своим порогом: если он top-1 и прошёл порог — рисунок неприличный.

## 4. Точность порта

Эталон считает во `float64`. Порт во `float32` отличается по интенсивности не более чем на ~1e-5
(тест `tests/test_gpu_raster.py` сверяет float32-реализацию на torch с допуском 1e-4). Для приёмки порта:
прогнать каждый `tests/vectors/*.json` (`input` → нормализация → растр) и сравнить:

- `simplified` — **точное** совпадение целых координат;
- растр с `<name>.npy` (`float32 [N, N]`) — `max |Δ| < 1e-3`.

Векторы `*_n32` проверяют другие параметры (`size 32, pad 1, line_width 1.5`) — параметры в `params`.

## 5. Набросок на C# (не компилировался — ориентир)

```csharp
// strokes: List<List<Vector2>> в координатах с осью Y вниз (см. §0).
static List<List<Vector2Int>> Normalize(List<List<Vector2>> strokes, float eps = 2f)
{
    var pts = strokes.Where(s => s.Count > 0).ToList();
    var res = new List<List<Vector2Int>>();
    if (pts.Count == 0) return res;
    double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
    foreach (var s in pts) foreach (var p in s) { minX = Math.Min(minX, p.x); maxX = Math.Max(maxX, p.x); minY = Math.Min(minY, p.y); maxY = Math.Max(maxY, p.y); }
    double ext = Math.Max(maxX - minX, maxY - minY), sc = ext > 0 ? 255.0 / ext : 1.0;
    foreach (var s in pts)
    {
        var q = Dedupe(s.Select(p => ((p.x - minX) * sc, (p.y - minY) * sc)).ToList());
        q = Rdp(q, eps);
        var r = Dedupe(q.Select(p => (Math.Floor(p.Item1 + 0.5), Math.Floor(p.Item2 + 0.5))).ToList());
        res.Add(r.Select(p => new Vector2Int((int)p.Item1, (int)p.Item2)).ToList());
    }
    return res;
}

static float[] Rasterize(List<List<Vector2Int>> strokes, int n = 64, float pad = 2f, float lineWidth = 2f)
{
    var img = new float[n * n];
    if (strokes.Count == 0) return img;
    int minX = int.MaxValue, minY = int.MaxValue, maxX = int.MinValue, maxY = int.MinValue;
    foreach (var s in strokes) foreach (var p in s) { /* min/max */ }
    float cx = (minX + maxX) * 0.5f, cy = (minY + maxY) * 0.5f, f = (n - 2 * pad) / 255f, r = lineWidth * 0.5f;
    var best = Enumerable.Repeat(float.MaxValue, n * n).ToArray();
    foreach (var s in strokes)
        for (int k = 0; k < Math.Max(1, s.Count - 1); k++)
        {
            var a = ToPx(s[k]); var b = s.Count > 1 ? ToPx(s[k + 1]) : a;   // ToPx: n/2 + (v − c)·f
            // обойти пиксели рамки отрезка ± (r + 1), best[i] = min(best[i], DistToSegment(center, a, b))
        }
    for (int i = 0; i < img.Length; i++) img[i] = Mathf.Clamp01(r + 0.5f - best[i]);
    return img;   // row-major, строка 0 — верх
}
```
