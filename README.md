# Lensless imaging

Этот репозиторий содержит реализацию методов реконструкции для задачи lensless computational imaging на датасете `DigiCam-Mirflickr-MultiMask-10K`.

## Что реализовано

- модели:
  - `ADMM-100`;
  - `Unrolled ADMM-20`;
  - `LeADMM-5 Pre`;
  - `LeADMM-5 Post`;
  - `LeADMM-5 Pre+Post`.
- функции потерь:
  - `MSE`;
  - `LPIPS`;
  - их взвешенная сумма в `ReconstructionLoss`.
- метрики:
  - `MSE`;
  - `PSNR`;
  - `SSIM`;
  - `LPIPS`.
- обучение через `Hydra`.
- логирование в `Comet ML`.
- инференс на пользовательском датасете через `CustomDirDataset`.
- отдельный скрипт для подсчета метрик по сохраненным реконструкциям.
- analysis:
  - qualitative;
  - quantitative;
  - speed comparison.

## Структура

Основные папки и файлы:

- `train.py` - запуск обучения.
- `inference.py` - запуск инференса на датасете.
- `calculate_metrics.py` - отдельный подсчет метрик по папкам с картинками.
- `src/model/` - реализации `LeADMM`, `ModularLeADMM` и `DRUNet`.
- `src/loss/` - функции потерь.
- `src/metrics/` - метрики и tracker.
- `src/datasets/` - датасеты и collate-функции.
- `src/trainer/` - trainer и inferencer.
- `src/configs/` - Hydra-конфиги.
- `analysis/` - код для qualitative / quantitative analysis.
- `analysis.ipynb` - ноутбук с анализом.
- `demo.ipynb` - demo для инференса на пользовательском датасете.

## Установка

Рекомендуемая версия Python: `3.12.5` (модель обучалось и тестировалась на ней).

Сначала можно клонировать репозиторий:

```bash
git clone https://github.com/Zagonov/lensless-imaging.git
cd lensless-imaging
```

После этого установить зависимости:

```bash
pip install -r requirements.txt
```

## Данные

В основном эксперименте используются:

- train: `DigiCam-Mirflickr-MultiMask-10K/train`
- test / final eval: `DigiCam-Mirflickr-MultiMask-10K/test`

По умолчанию датасет скачивается через `datasets` и кэшируется в `.cache`

Конфиг датасета:

```text
src/configs/datasets/digicam_mirflickr.yaml
```

## Обучение

### Образец конфига:

```text
src/configs/baseline.yaml
```

`baseline.yaml` повторяет `unrolled_admm_20.yaml`, сделан в качестве образца.

Запуск:

```bash
python train.py --config-name baseline
```

### Отдельные эксперименты

#### ADMM-100

Конфиг:

```text
src/configs/admm_100.yaml
```

Запуск:

```bash
python train.py --config-name admm_100
```

Это не обучаемая модель, поэтому в этом конфиге делается один шаг, чтобы прогнать evaluation, сохранить логи и чекпоинт в том же пайплайне.

#### Unrolled ADMM-20

Конфиг:

```text
src/configs/unrolled_admm_20.yaml
```

Запуск:

```bash
python train.py --config-name unrolled_admm_20
```

#### LeADMM-5 Pre

Конфиг:

```text
src/configs/leadmm5_pre.yaml
```

Запуск:

```bash
python train.py --config-name leadmm5_pre
```

#### LeADMM-5 Post

Конфиг:

```text
src/configs/leadmm5_post.yaml
```

Запуск:

```bash
python train.py --config-name leadmm5_post
```

#### LeADMM-5 Pre+Post

Конфиг:

```text
src/configs/leadmm5_pre_post.yaml
```

Запуск:

```bash
python train.py --config-name leadmm5_pre_post
```

## Логи и чекпоинты

По умолчанию логирование идет в `Comet ML` через:

```text
src/configs/writer/cometml.yaml
```

В логах сохраняются:

- train / test loss;
- `MSE`, `PSNR`, `SSIM`, `LPIPS`;
- изображения реконструкций;
- `grad_norm`;
- `learning_rate`.

Чекпоинты обучения сохраняются в:

```text
saved/<run_name>/
```

Если в конфиге задан monitor, то после обучения сохраняется лучший чекпоинт:

```text
saved/<run_name>/model_best.pth
```

## Как воспроизвести модель

Для любого из экспериментов схема одна и та же:

1. установить зависимости:

```bash
pip install -r requirements.txt
```

2. запустить обучение нужного конфига:

```bash
python train.py --config-name leadmm5_pre_post
```

3. после обучения использовать лучший чекпоинт:

```text
saved/leadmm5_pre_post/model_best.pth
```

## Скачивание чекпоинтов

### ADMM-100

```bash
mkdir -p saved/admm_100
wget -O saved/admm_100/admm_100_best.pth https://huggingface.co/Lunfus/admm_100/resolve/main/admm_100_best.pth
```

### Unrolled ADMM-20

```bash
mkdir -p saved/unrolled_admm_20
wget -O saved/unrolled_admm_20/unrolled_admm_20_best.pth https://huggingface.co/Lunfus/unrolled_admm_20/resolve/main/unrolled_admm_20_best.pth
```

### LeADMM-5 Pre

```bash
mkdir -p saved/leadmm5_pre
wget -O saved/leadmm5_pre/leadmm5_pre_best.pth https://huggingface.co/Lunfus/leadmm5_pre/resolve/main/leadmm5_pre_best.pth
```

### LeADMM-5 Post

```bash
mkdir -p saved/leadmm5_post
wget -O saved/leadmm5_post/leadmm5_post_best.pth https://huggingface.co/Lunfus/leadmm5_post/resolve/main/leadmm5_post_best.pth
```

### LeADMM-5 Pre+Post

```bash
mkdir -p saved/leadmm5_pre_post
wget -O saved/leadmm5_pre_post/leadmm5_pre_post_best.pth https://huggingface.co/Lunfus/leadmm_5_pre_post/resolve/main/leadmm5_pre_post_best.pth
```

## Inference

Основная логика инференса вынесена в:

```text
inference.py
```

По умолчанию для инференса используется:

```text
src/configs/inference.yaml
```

и кастомный датасет:

```text
src/configs/datasets/custom_dir.yaml
```

Формат пользовательского датасета:

```text
data/custom
├── lensless
│   ├── ImageID1.png
│   ├── ImageID2.png
│   └── ...
├── masks
│   ├── ImageID1.npy
│   ├── ImageID2.npy
│   └── ...
└── lensed
    ├── ImageID1.png
    ├── ImageID2.png
    └── ...
```

Папка `lensed` может отсутствовать. Тогда реконструкции просто сохраняются без подсчета метрик.

Пример инференса для `LeADMM-5 Pre+Post`:

```bash
python inference.py model=leadmm5_pre_post inferencer.from_pretrained=saved/leadmm5_pre_post/leadmm5_pre_post_best.pth datasets.custom.data_dir=data/custom inferencer.save_path=saved/inference_test
```

Результаты сохраняются в:

```text
saved/inference_test/custom/
```

## Подсчет метрик

Для этого есть отдельный скрипт:

```text
calculate_metrics.py
```

Пример запуска:

```bash
python calculate_metrics.py --data-dir data/custom --predictions-dir saved/inference_test/custom
```

На выходе печатаются:

- `num images`
- `MSE`
- `PSNR`
- `SSIM`
- `LPIPS`

## Demo

Для демонстрации работы используется:

```text
demo.ipynb
```

В ноутбуке делается следующее:

1. устанавливаются зависимости для demo;
2. задается конфиг модели;
3. скачивается нужный чекпоинт;
4. по пользовательской ссылке на Google Drive скачивается `.zip` датасет;
5. архив распаковывается;
6. запускается `inference.py`;
7. показываются `original / lensless / reconstruction`;
8. при наличии `lensed` запускается `calculate_metrics.py`.

В ноутбуке можно выбрать одну из 5 моделей:

- `admm_100`
- `unrolled_admm_20`
- `leadmm5_pre`
- `leadmm5_post`
- `leadmm5_pre_post`

## Анализ

### Qualitative analysis

Основные функции находятся в:

- `analysis/qualitative.py`
- `analysis/common.py`

Пример использования:

```python
from analysis import show_qualitative_examples, show_error_maps

show_qualitative_examples(num_examples=3)
show_error_maps(num_examples=3)
```

### Quantitative analysis

Основные функции находятся в:

- `analysis/quantitative.py`

В нем считаются:

- средние `MSE`, `PSNR`, `SSIM`, `LPIPS` по моделям;
- распределения `PSNR` и `LPIPS`;
- скорость реконструкции:
  - `sec_per_image`
  - `images_per_sec`

Пример использования:

```python
from analysis import show_metric_table, show_metric_distributions, show_speed_table

show_metric_table(num_samples=None)
show_metric_distributions(num_samples=None)
show_speed_table(num_samples=20, warmup=2)
```

Готовый ноутбук с анализом:

```text
analysis.ipynb
```
