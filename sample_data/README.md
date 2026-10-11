# ДЗ 1 — примеры данных и ожидаемый вывод

Отдельных файлов здесь нет: примеры — кадры лекций, они уже лежат в репозитории курса.
Прогоните свои команды на них и сверьте числа с таблицей: расхождение больше допуска —
ошибка в функции, а не «другая машина».

| Кадр | Путь в репозитории курса | Что известно точно |
|------|--------------------------|--------------------|
| здание ИУ | `lectures/L05-filtering/demo/iu-building.jpg` | 1920 × 1279 |
| мяч при свете дня / под лампой | `lectures/L03-color-histograms/demo/ball-day.jpg`, `ball-lamp.jpg` | 960 × 640; мяч нарисован: центр (190, 560), радиус 58 px — эталон для recall / precision |
| плакат под углом | `lectures/L04-geometry/demo/poster-angle.jpg` | 1920 × 1279; углы (560, 330), (1250, 220), (1290, 1010), (600, 840) |
| плакат «в лоб» | `lectures/L04-geometry/demo/poster-gt.png` | 640 × 905 — эталон выпрямления |
| фон для хромакея | `lectures/L04-geometry/demo/road-cam.jpg` | любой другой кадр тоже подойдёт |

Ниже `IU = lectures/L05-filtering/demo/iu-building.jpg`, `L3 = lectures/L03-color-histograms/demo`,
`L4 = lectures/L04-geometry/demo`. Числа получены эталонной реализацией на OpenCV 4.14.0.94;
допуск — последний знак, если не сказано иное.

## Цветокоррекция

```
python photolab.py gamma IU --gamma 0.5 -o out/g05.jpg --hist out/g05_hist.png
  before: mean 134.6, median 123, p1 3, p99 254, clipped white 0.33 %
  after:  mean 174.3, median 177, p1 24, p99 254, clipped white 0.72 %
python photolab.py gamma IU --gamma 2.0 -o out/g20.jpg
  after:  mean 95.0, median 60, p1 0, clipped black 3.78 %

python photolab.py wb L3/ball-lamp.jpg --method grey-world -o out/wb_gw.jpg
  gains B, G, R = 0.999, 1.028, 0.975 · means before 61.3 / 59.6 / 62.8 → after 61.3 / 61.2 / 61.2
python photolab.py wb L3/ball-lamp.jpg --method white-patch -o out/wb_wp.jpg
  gains B, G, R = 2.318, 2.217, 2.008 · means after 142.1 / 132.0 / 126.0
  (кадр «под лампой» почти нейтрален и вдвое темнее: grey-world его почти не трогает,
   white-patch тянет 99-й процентиль к 255 и осветляет вдвое — оба поведения правильные)

python photolab.py clahe IU --clip 2 --tiles 8 -o out/clahe.jpg --compare out/clahe_cmp.jpg
  тени (L ≤ 20-й процентиль): σ исходный 21.1 · equalizeHist 14.8 · CLAHE 32.3
  света (L ≥ 80-й процентиль): среднее 240.7 · 230.3 · 230.5
  (equalizeHist сжимает тени, CLAHE растягивает; света оба методы притемняют одинаково)
```

## Хромакей

```
python photolab.py chromakey L3/ball-lamp.jpg --bg L4/road-cam.jpg --lo 140,121,40 --hi 168,230,255 \
    -o out/ck.jpg --mask out/ck_mask.png
  bbox (133, 502, 115 × 116), area 9996, components 1, raw_components 1, holes 0, mask 1.63 %
  против эталона мяча: recall 94.6 %, precision 99.9 %          (те же числа, что в занятии 3)
python photolab.py chromakey L3/ball-day.jpg --bg L4/road-cam.jpg --sample 160,530,60,60 -o out/ck2.jpg
  границы с прямоугольника: lo (140, 88, 40), hi (166, 228, 255); area 10581, holes 0
```

Recall и precision команда не печатает — эталон есть только у кадров курса; посчитайте их
в отчёте по маске `out/ck_mask.png` и кругу (190, 560, 58): три строки NumPy.

## Геометрия

```
python photolab.py rotate IU --deg 30 -o out/rot30.jpg
  1920 × 1279 → 2302 × 2068, пустого холста 48.4 %                    (слайд 19 L4)

python photolab.py resize IU --scale 0.25 --report -o out/small.jpg
  480 × 320; PSNR к Lanczos: naive 27.9 дБ · blur 36.0 · area 37.2     (демо 2.4 L4: 27.9 / 36.7 при σ 1.5 / 37.2)
python photolab.py resize IU --scale 2 -o out/big.jpg
  3840 × 2558 (INTER_CUBIC)

python photolab.py rectify L4/poster-angle.jpg --corners 1290,1010,560,330,600,840,1250,220 --aspect 0.707 -o out/front.jpg
  углы упорядочены: (560, 330), (1250, 220), (1290, 1010), (600, 840)
  стороны в кадре: top 698.7, right 791.0, bottom 710.6, left 511.6 px → выход 711 × 1006 (W/H 0.707)
python photolab.py rectify L4/poster-angle.jpg --corners 560,330,1250,220,1290,1010,600,840 --size 640x905 -o out/front640.jpg
  совпало с poster-gt.png после бинаризации: 99.1 % пикселей           (слайд 5 L4, занятие 4)
```

## Контактный лист

```
python photolab.py sheet IU L3/ball-day.jpg L4/poster-angle.jpg L4/road-cam.jpg L4/text-page.png --cols 3 -o out/sheet.jpg
  n 5, cols 3, лист 992 × 556 (ячейки 320 × 240, подпись 26 px, поля 8 px)
```
