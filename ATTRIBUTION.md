# Атрибуция данных

Модель обучается на наборе данных **Quick, Draw!**.

- Источник: Google, Inc. — «Quick, Draw!» Dataset
- Набор данных: https://github.com/googlecreativelab/quickdraw-dataset
- Лицензия: Creative Commons Attribution 4.0 International (CC BY 4.0) — https://creativecommons.org/licenses/by/4.0/
- Данные изменены: использованы для обучения модели (отбор классов и рисунков с `recognized: true`,
  нормализация штрихов, растеризация, аугментации).

Строка для титров игры / страницы «О программе»:

> Распознавание рисунков обучено на данных «Quick, Draw!» © Google, Inc., CC BY 4.0
> (github.com/googlecreativelab/quickdraw-dataset). Данные изменены: использованы для обучения модели.

Собственные рисунки (`data/custom/`, класс `rude` и дополнительные рисунки заданий) собраны командой
через редактор этого проекта и к Quick Draw не относятся.

# onnxruntime-web (playground)

«Угадайка» (`playground/`) запускает модель в браузере через **onnxruntime-web** (пакет
`onnxruntime-web`, файлы вшиты в `playground/vendor/` скриптом `scripts/fetch_ort.py`, версия — в
`playground/vendor/VERSION`).

- Проект: ONNX Runtime — https://github.com/microsoft/onnxruntime
- Copyright (c) Microsoft Corporation
- Лицензия: MIT — https://github.com/microsoft/onnxruntime/blob/main/LICENSE

```
MIT License. Copyright (c) Microsoft Corporation.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
```
