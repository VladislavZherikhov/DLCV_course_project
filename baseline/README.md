# D-Fire: классификация огня и дыма (baseline ResNet18)

4 класса по YOLO-разметке D-Fire (`0 = smoke`, `1 = fire`):

| id | класс | правило |
|---|---|---|
| 0 | `none` | txt пустой или отсутствует |
| 1 | `smoke` | есть только `0` |
| 2 | `fire` | есть только `1` |
| 3 | `fire_smoke` | есть и `0`, и `1` |

Метрики (val и test): accuracy, precision / recall / F1 / ROC AUC (one-vs-rest) в вариантах **macro** и **weighted**,
по каждому классу (+ average precision), матрица ошибок, ROC и PR кривые.

## Структура

| файл | что делает |
|---|---|
| `config.py` | все параметры; пути автоматически переключаются между Kaggle и локальной машиной |
| `main.py` | точка входа: данные → модель → обучение → оценка best.pt на val и test |
| `data.py` | txt → 4 класса, сплит train/val, аугментации, Dataset/DataLoader |
| `model.py` | ResNet из torchvision с новой головой |
| `train.py` | эпоха обучения (лог каждые N батчей), предсказание, `fit` (best/last, early stopping, log.csv) |
| `utils.py` | seed, логгер, метрики, графики, отчёты |
| `predict.py` | отдельный тест сохранённой модели / предсказание на своих фото |
| `check_data.py` | проверка датасета: распределение классов + `data_check.png` с боксами class id 0 / 1 |

## Kaggle (код через GitHub)

Настройки ноутбука: **Add Input** → датасет D-Fire; **Accelerator: GPU**; **Internet: On**.

```python
# 1. GPU
!nvidia-smi
```
```python
# 2. Код из GitHub (каждый запуск берёт свежую версию)
REPO_URL = 'https://github.com/VladislavZherikhov/DLCV_course_project.git'
%cd /kaggle/working
!rm -rf repo && git clone -q -b baseline {REPO_URL} repo
%cd /kaggle/working/repo/baseline
!git log -1 --oneline
!pip install -q omegaconf
```
```python
# 3. Проверка датасета: class id 0 должен быть дымом, 1 — огнём
!python check_data.py
from IPython.display import Image, display
display(Image('data_check.png'))
```
```python
# 4. Debug-прогон (1–2 минуты)
!python -u main.py general.debug=true
```
```python
# 5. Обучение + тест
EXP_NAME = 'resnet18_224'
!python -u main.py general.exp_name={EXP_NAME}
```
```python
# 6. Результаты
import pandas as pd
EXP = f'/kaggle/working/outputs/{EXP_NAME}'
print(open(f'{EXP}/test_report.txt').read())
for name in ['training_curves.png', 'test_confusion_matrix.png', 'test_confusion_matrix_norm.png',
             'test_roc_pr_curves.png', 'test_errors.png']:
    print(name); display(Image(f'{EXP}/{name}'))
display(pd.read_csv(f'{EXP}/log.csv'))
display(pd.read_csv('/kaggle/working/outputs/results.csv'))
```

Ячейки 1–4 прогнать вручную, затем **Save Version → Save & Run All (Commit)** — обучение идёт в фоне.
Результаты — в версии ноутбука, вкладка **Output** → `outputs/resnet18_224.zip`.
После правок кода: `git commit` + `git push` локально, в Kaggle перезапустить ячейку 2.

Датасет ищется рекурсивно внутри `/kaggle/input` — подходят раскладки `train/images + train/labels`
и `images/train + labels/train`. Если найдётся не то — `paths.data_root=/kaggle/input/...`.

## Запуск из консоли

```bash
pip install -r requirements.txt
python main.py                                    # параметры из config.py
python main.py general.debug=true                 # 128 картинок на сплит, 1 эпоха — проверка
python main.py general.exp_name=r18_lr1e-4 training.lr=1e-4 training.epochs=20
```

Любой параметр из `config.py` переопределяется как `секция.ключ=значение`. Опечатка в ключе — ошибка.

## Что сохраняется

`outputs/<exp_name>/` (и то же самое в `outputs/<exp_name>.zip`):

| файл | содержимое |
|---|---|
| `train.log` | полный лог: система, конфиг, распределение классов, каждые N батчей (loss, lr, img/s, память GPU), по эпохам — метрики train/val и per-class таблица, финальный тест |
| `log.csv` | все метрики по эпохам (train/val, per-class precision/recall/f1, lr, время, память) |
| `training_curves.png` | loss, accuracy/F1 train vs val, precision/recall/F1, ROC AUC, F1 по классам, lr |
| `best.pt`, `last.pt` | чекпоинты (веса + конфиг + классы) |
| `config.yaml`, `summary.json` | с какими параметрами запускали и итоговые метрики |
| `class_distribution.csv`, `labels.csv` | распределение классов и какая картинка в какой сплит попала |
| `{val,test}_report.txt` | все метрики + per-class таблица + матрица ошибок в читаемом виде |
| `{val,test}_metrics.json`, `{val,test}_per_class.csv`, `{val,test}_confusion_matrix.csv` | то же в машинно-читаемом виде |
| `{val,test}_confusion_matrix.png`, `..._norm.png` | матрица ошибок (штуки и доли) |
| `{val,test}_roc_pr_curves.png` | ROC и Precision-Recall кривые по классам |
| `{val,test}_errors.png` | 24 самые уверенные ошибки модели |
| `{val,test}_predictions.csv` | предсказание и вероятности для каждой картинки |

Плюс `outputs/results.csv` — одна строка на эксперимент для сравнения.

## Локальная проверка обученной модели

```bash
# полная оценка на test (нужен скачанный D-Fire), результаты в outputs/<exp>/local_eval/
python predict.py --ckpt outputs/resnet18_224/best.pt --data_root C:/data/D-Fire --split test

# свои фото
python predict.py --ckpt outputs/resnet18_224/best.pt --images my_photos/
```
