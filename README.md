# Сегментация хвоста кота

Инстанс-сегментация кошачьих хвостов с использованием YOLOv11n-seg и transfer learning на небольшом датасете.

## Задача

Детекция и сегментация хвостов котов на изображениях с помощью дообученной модели YOLO. В датасете COCO нет класса "хвост", поэтому мы вручную разметили 43 изображения, сгенерировали 50 синтетических и обучили свою модель.

## Результаты

| Метрика          | Значение |
|------------------|----------|
| mAP50 (box)      | 0.682    |
| mAP50-95 (box)   | 0.469    |
| mAP50 (mask)     | 0.775    |
| mAP50-95 (mask)  | 0.537    |

Обучено на 43 реальных + 50 синтетических = 93 изображения. Валидация на 13 изображениях.

## Структура проекта

```
.
├── README.md
├── requirements.txt
├── train.py                        # Точка входа для обучения
├── predict.py                      # CLI инференс
├── yolo11n-seg.pt                  # Предобученные веса YOLO (COCO)
├── weights/
│   └── best.pt                     # Обученные веса модели
├── app/
│   └── app.py                      # Gradio веб-демо
├── scripts/
│   ├── convert_labelme_to_yolo.py  # LabelMe JSON → YOLO .txt
│   ├── split_dataset.py            # Разбиение на train/val (80/20)
│   ├── augment_synthetic.py        # Генерация синтетических данных
│   └── verify_labels.py            # Визуальная проверка разметки
├── data/
│   ├── raw/                        # Оригинальные изображения + JSON от LabelMe
│   ├── images/{train,val}/         # Разделённые изображения
│   ├── labels/{train,val}/         # YOLO-разметка (сегментация)
│   └── dataset.yaml                # Конфиг датасета для YOLO
├── runs/                           # Результаты обучения и веса
└── docs/
    ├── ARCHITECTURE.md             # Архитектура и решения
    ├── DATA_PIPELINE.md            # Пайплайн подготовки данных
    ├── TRAINING.md                 # Стратегия обучения и гиперпараметры
    ├── AUGMENT_SYNTHETIC.md        # augment_synthetic.py построчно
    ├── CONVERT_LABELME_TO_YOLO.md  # convert_labelme_to_yolo.py построчно
    ├── SPLIT_DATASET.md            # split_dataset.py построчно
    ├── VERIFY_LABELS.md            # verify_labels.py построчно
    ├── TRAIN.md                    # train.py построчно
    ├── PREDICT.md                  # predict.py построчно
    └── APP.md                      # app/app.py построчно
```

## Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Примечание:** Используется `opencv-python-headless` для избежания конфликтов Qt с LabelMe.

## Пайплайн данных

### 1. Сбор изображений
Поместите фотографии котов (с видимыми хвостами) в `data/raw/`.

### 2. Разметка в LabelMe
```bash
labelme data/raw/ --output data/raw/ --labels tail
```
Обведите контур хвоста инструментом полигон. Метка — `tail`.

### 3. Конвертация разметки
```bash
python scripts/convert_labelme_to_yolo.py
```
Конвертирует LabelMe JSON → YOLO segment `.txt` (нормализованные координаты полигона).

### 4. Разбиение датасета
```bash
python scripts/split_dataset.py
```
Разбиение 80/20 на train/val. Копирует изображения и разметку в `data/images/` и `data/labels/`.

### 5. Генерация синтетических данных (опционально)
```bash
python scripts/augment_synthetic.py
```
Вырезает хвосты из реальных изображений и вставляет на сгенерированные фоны со случайными аугментациями. Добавляет ~50 дополнительных тренировочных изображений.

### 6. Проверка разметки
```bash
python scripts/verify_labels.py
```
Рисует маски поверх изображений в `data/verify/` для визуальной проверки.

## Обучение

```bash
python train.py
```

### Конфигурация
| Параметр   | Значение | Описание                             |
|------------|----------|--------------------------------------|
| Model      | YOLOv11n-seg | Предобучена на COCO             |
| imgsz      | 800      | Размер входного изображения          |
| batch      | 8        | Размер батча                         |
| epochs     | 150      | Макс. эпох (early stopping включён)  |
| freeze     | 10       | Замороженные слои backbone           |
| patience   | 20       | Терпение early stopping              |
| optimizer  | AdamW    | Оптимизатор                          |
| lr0        | 0.01     | Начальный learning rate              |
| lrf        | 0.01     | Финальный learning rate (множитель)  |
| cos_lr     | True     | Косинусное расписание lr             |

### Transfer Learning
Backbone (первые 10 слоёв) заморожен — обучается только голова детекции/сегментации. Это предотвращает переобучение на маленьком датасете, используя признаки, выученные на COCO.

### Аугментации
Поворот (±15°), сдвиг, масштабирование, горизонтальное отражение, HSV jitter, mosaic, mixup, random erasing — всё задано явно в `train.py`.

Лучшие веса: `weights/best.pt`

## Инференс

### CLI
```bash
# Одно изображение
python predict.py --source photo.jpg --conf 0.3

# Папка
python predict.py --source test_photos/

# Видео
python predict.py --source video.mp4

# Свои веса
python predict.py --source photo.jpg --weights path/to/best.pt
```
Результаты сохраняются в `runs/predict/result/`.

### Веб-приложение (Gradio)
```bash
# По умолчанию — weights/best.pt, порт 7860
python app/app.py

# Свои веса
python app/app.py --weights path/to/best.pt

# Свой порт
python app/app.py --port 8080
```
Откройте http://localhost:<порт> (по умолчанию 7860) — загрузите фото кота, настройте порог уверенности, увидите сегментированный хвост.

Если хвост не обнаружен, на изображении отобразится надпись "No tail detected".

## Стек технологий

- **Модель:** Ultralytics YOLOv11n-seg
- **Разметка:** LabelMe (полигоны)
- **Аугментация:** Albumentations + встроенная в YOLO
- **Демо:** Gradio
- **Язык:** Python 3.10+
