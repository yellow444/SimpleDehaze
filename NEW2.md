Да, нужны **архивы чистых и покрытых дымкой изображений**, но для исследовательской работы над A²CR этого недостаточно. Нам нужны четыре типа данных:

1. **Реальные парные фотографии**: одна и та же сцена без дымки и с настоящей искусственно созданной дымкой.
2. **Синтетические пары**: чистое изображение и его версия с математически добавленной дымкой.
3. **Чистые RGB-изображения с картами глубины**: чтобы самим генерировать дымку и точно знать (t), (A), глубину и уровень шума.
4. **Реальные фото без эталона**: чтобы проверить, не работает ли метод хорошо только на лабораторных данных.

Для предложенного training-free метода большие обучающие датасеты не обязательны. Данные нужны прежде всего для **проверки формул, настройки нескольких общих параметров и независимого тестирования**.

> **Статус реализации, 29 июля 2026.** Практический минимум из этого документа выполнен:
> локально подготовлены I-HAZE, O-HAZE, Dense-Haze, NH-HAZE и полный DIODE validation;
> архив DIODE проверен по опубликованному MD5, extraction содержит 771 RGB/depth/mask triple,
> а SHA-256 inventory — 2625 файлов. Детерминированный manifest выбирает 500 DIODE-кадров
> (250 indoor/250 outdoor), делит шесть физических сцен на 295/104/101 без leakage и задаёт
> 22 500 streaming-рецептов. Полный контролируемый прогон дал 112 500 строк без сбоев;
> четыре real-paired test-split дали 364/364 успешных результата с настоящим LPIPS/AlexNet.
> Команды, точные метрики и ограничения зафиксированы в
> `SimpleDeHaze/docs/research/a2cr-data-protocol.md` и `REPRODUCIBILITY.md`.
>
> Большие/опциональные RESIDE, CARLA-Haze, LMHaze, Foggy Driving и DIODE train намеренно не
> скачивались: ниже этот документ сам относит их к следующему этапу. Внешняя полная BCCR
> reference-реализация тоже не подменяется похожим локальным методом: B3 проверяет только
> boundary constraint, а внешние результаты импортируются через `--evaluate`. Реальные наборы
> уже использовались при разработке, поэтому текущий real-paired результат не называется blind test.

# Что скачать в первую очередь

## 1. Реальные парные наборы — основной итоговый тест

Это самые важные архивы для статьи. В них дымка действительно создавалась физически, а не накладывалась формулой.

| Набор          | Содержимое              | Для чего нужен                                         | Ссылка                                                                                    |
| -------------- | ----------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| **I-HAZE**     | 35 indoor-пар, 312 МБ   | Помещения, цветовые мишени, близкие объекты            | [Официальная страница и загрузка](https://data.vision.ee.ethz.ch/cvl/ntire18/i-haze/)     |
| **O-HAZE**     | 45 outdoor-пар, 547 МБ  | Обычная равномерная дымка на улице                     | [Официальная страница и загрузка](https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/)     |
| **Dense-Haze** | 33 outdoor-пары, 245 МБ | Очень плотная равномерная дымка                        | [Официальная страница и загрузка](https://data.vision.ee.ethz.ch/cvl/ntire19/dense-haze/) |
| **NH-HAZE**    | 55 outdoor-пар, 330 МБ  | Неоднородная дымка: локальные облака, разная плотность | [Официальная страница и загрузка](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/)    |

I-HAZE содержит реальные пары indoor-сцен, снятые при одинаковых условиях освещения, с физически созданной дымкой и цветовой мишенью. O-HAZE аналогично содержит 45 outdoor-сцен. ([Data Vision][1])

Dense-Haze специально предназначен для плотной однородной дымки и содержит 33 пары, а NH-HAZE — 55 сцен с настоящей неоднородной дымкой. ([Data Vision][2])

Суммарно эти четыре архива занимают приблизительно **1,4 ГБ**. Это первый обязательный пакет.

### Как использовать

Эти четыре набора лучше сразу объявить **закрытыми тестовыми**:

* не подбирать по ним коэффициенты;
* не выбирать по ним лучший вариант алгоритма;
* не менять параметры после просмотра результатов;
* один раз зафиксировать конфигурацию и прогнать все наборы.

Именно эти результаты затем идут в главные таблицы статьи.

---

## 2. LMHaze — большой реальный парный набор

LMHaze содержит 5040 пар реальных изображений с несколькими уровнями дымки. В публичной версии изображения приведены к разрешению `1200 × 800`; официальное разбиение включает 3925 пар для train и 1115 для test. ([GitHub][3])

Официальный репозиторий:

[LMHaze на GitHub](https://github.com/wangzrk/LMHaze)

Прямые ссылки авторов:

* [LMHaze train — 3925 пар, Google Drive](https://drive.google.com/file/d/10BiQF9oTexwo3EdRKeyqhY39Q_qAAAc2/view?usp=sharing)
* [LMHaze test — 1115 пар, Google Drive](https://drive.google.com/file/d/1V-PwHbLfNgg1nF64bPK9HHUlc93bDSGs/view?usp=sharing)

### Для чего использовать

* `train` — для первоначальной проверки устойчивости или выбора общих параметров;
* `test` — как независимый большой реальный benchmark;
* отдельно сравнить indoor и outdoor;
* отдельно сравнить уровни плотности дымки.

Для первой реализации A²CR LMHaze не обязателен. Сначала достаточно четырёх маленьких реальных наборов, поскольку они проще в обработке и широко используются в литературе.

---

# 3. RESIDE — стандартный синтетический benchmark

Официальная страница:

[RESIDE: A Benchmark for Single Image Dehazing](https://sites.google.com/view/reside-dehaze-datasets)

RESIDE состоит из нескольких поднаборов:

| Поднабор | Что в нём                                    | Нужно ли сейчас             |
| -------- | -------------------------------------------- | --------------------------- |
| **ITS**  | Синтетические indoor-пары                    | Позже                       |
| **OTS**  | Большой outdoor-набор                        | Позже                       |
| **SOTS** | Синтетический тест с эталонами               | **Да**                      |
| **RTTS** | Реальные изображения с дымкой без чистого GT | **Да, для реального теста** |
| **HSTS** | Смешанный субъективный тест                  | Опционально                 |

Официальная страница содержит ссылки на ITS, OTS, SOTS, RTTS и HSTS, а также эталонный код расчёта PSNR и SSIM. ([Google Sites][4])

На первом этапе скачивайте только:

* **SOTS** — объективная проверка PSNR/SSIM;
* **RTTS** — визуальная проверка реальных сцен без ground truth;
* при необходимости HSTS.

Полные ITS и OTS нужны преимущественно для обучения нейросетей. Для аналитического A²CR загружать десятки тысяч обучающих изображений сразу нет смысла.

---

# 4. RGB + depth: главное для проверки нового метода

Для A²CR особенно важны не готовые пары, а **чистые изображения с истинной картой глубины**.

Почему:

[
I(x)=J(x)t(x)+A(1-t(x))+n(x),
]

[
t(x)=e^{-\beta d(x)}.
]

Если известны чистое изображение (J) и глубина (d), мы сами задаём:

* atmospheric light (A);
* плотность дымки (\beta);
* точную карту transmission (t);
* шум (n);
* ошибку оценки (\hat t);
* ошибку atmospheric light (\hat A).

Это позволяет проверять не только качество итоговой картинки, но и **правильность внутренних формул A²CR**.

## DIODE — лучший исходный набор для начала

Официальная страница:

[DIODE: Dense Indoor and Outdoor Depth Dataset](https://diode-dataset.org/)

Набор включает RGB, плотные карты глубины и validity masks как для indoor, так и для outdoor. В validation-разбиении 771 изображение: 325 indoor и 446 outdoor. Архив validation занимает 2,6 ГБ. ([diode-dataset.github.io][5])

Прямая официальная загрузка:

[DIODE Validation — 2,6 ГБ](http://diode-dataset.s3.amazonaws.com/val.tar.gz)

MD5, указанный авторами:

```text
5c895d09201b88973c8fe4552a67dd85
```

Полный train весит 81 ГБ, поэтому сначала он не нужен. ([diode-dataset.github.io][5])

### Рекомендуемый первый контролируемый эксперимент

Взять 500 RGB-D сцен и создать:

* 5 уровней дымки;
* 3 цвета atmospheric light: нейтральный, холодный, тёплый;
* 3 уровня шума.

Получится:

[
500 \times 5 \times 3 \times 3=22500
]

контролируемых тестовых изображений.

Для каждого сохраняются:

```text
clear.png
depth.exr или depth.npy
hazy.png
transmission.exr
metadata.json
```

Пример метаданных:

```json
{
  "source_dataset": "DIODE",
  "scene_id": "outdoor_scene_03",
  "clear_image": "clear/000123.png",
  "depth_map": "depth/000123.npy",
  "hazy_image": "hazy/000123_v017.png",
  "beta": 0.8,
  "airlight_rgb_linear": [0.82, 0.88, 0.95],
  "noise_model": "poisson_gaussian",
  "noise_sigma": 0.01,
  "random_seed": 17421
}
```

Главное — проводить синтез в **linear RGB**, а затем преобразовывать результат обратно в sRGB.

---

## Дополнительные RGB-D источники

### NYU Depth V2

Indoor RGB-D набор. Подойдёт для помещений, мебели, белых поверхностей и сложного искусственного освещения.

[Официальная страница NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

В наборе 1449 плотно размеченных RGB-depth пар; также доступны значительно более крупные последовательности. ([TensorFlow][6])

### Middlebury Stereo

Небольшие, но очень качественные сцены с disparity/depth ground truth:

[Middlebury Stereo Datasets](https://vision.middlebury.edu/stereo/data/)

На официальной странице доступны наборы 2001, 2003, 2005, 2006, 2014 и 2021 годов; изображения и disparity maps разрешено использовать и публиковать при корректном цитировании. ([Vision][7])

### KITTI Depth

Outdoor и дорожные сцены:

[KITTI Depth Benchmark](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php)

Официальный benchmark содержит более 93 тысяч depth maps, RGB и LiDAR-данные. Однако карты глубины сложнее использовать для точного синтеза дымки, поскольку исходный LiDAR разреженный. Для первого эксперимента DIODE удобнее. ([CVLibs][8])

---

# 5. CARLA-Haze — готовый контролируемый outdoor benchmark

Официальный проект:

[CARLA-Haze](https://leoxthomas.github.io/CARLA-Haze/)

CARLA-Haze опубликован на WACV 2026 и содержит:

* 10 000 high-resolution пар clean/hazy;
* 10 разных сценариев;
* 10 уровней дымки;
* зависимость дымки от расстояния;
* готовые train/validation/test splits. ([Leo Thomas Ramos][9])

Для A²CR он особенно полезен для проверки следующего утверждения:

> Чем меньше transmission и выше неопределённость, тем сильнее regularized recovery должен отличаться от обычного (1/t).

CARLA-Haze лучше использовать как **внешний synthetic benchmark**, а собственный DIODE-синтез — как контролируемый эксперимент для проверки отдельных математических свойств.

---

# 6. Реальные сцены без чистого эталона

Парные наборы необходимы для PSNR и SSIM, но реальные природные условия ими полностью не покрываются.

## RTTS

Доступен на странице RESIDE:

[RESIDE / RTTS](https://sites.google.com/view/reside-dehaze-datasets)

RTTS содержит реальные изображения с дымкой и аннотации объектов, но без соответствующих чистых кадров. Поэтому на нём оцениваются:

* визуальная естественность;
* количество артефактов;
* поведение на небе;
* цветовые сдвиги;
* downstream object detection;
* no-reference метрики.

## Foggy Driving и Foggy Cityscapes

Официальная страница:

[Foggy Cityscapes / Foggy Driving](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/)

На ней доступны:

* **Foggy Driving** — 101 реальная дорожная сцена с аннотациями;
* **Foggy Cityscapes** — синтетические версии Cityscapes;
* карты transmission;
* восстановленные карты глубины;
* три фиксированных уровня плотности дымки. ([people.ee.ethz.ch][10])

Прямая ссылка на Foggy Driving находится на странице проекта; архив занимает около 100 МБ. Основные Foggy Cityscapes изображения загружаются через Cityscapes из-за условий лицензии. ([people.ee.ethz.ch][10])

Этот набор позволяет проверить не только «красивее ли изображение», но и:

> Улучшилось ли после dehazing распознавание машин, людей и дорожных объектов?

---

# 7. ACDC — дополнительная проверка реальных условий

[ACDC Dataset](https://acdc.vision.ee.ethz.ch/)

ACDC содержит 4006 изображений в условиях:

* fog;
* rain;
* snow;
* nighttime.

Для каждого adverse-condition изображения имеется соответствующая сцена в нормальных условиях и семантическая разметка. ([acdc.vision.ee.ethz.ch][11])

ACDC не стоит использовать как главный источник PSNR, поскольку пары могут быть не идеально пиксельно совмещены. Он полезен для:

* качественного сравнения;
* семантической сегментации;
* проверки устойчивости к реальной погоде;
* проверки того, не портит ли dehazing сцены без классической однородной дымки.

---

# Что пока не скачивать

## HazeSpace2M

[Официальный HazeSpace2M](https://github.com/tanvirnwu/HazeSpace2M_ACMM_2024)

Это очень большой набор:

* Outdoor — 269 ГБ;
* Street — 295 ГБ;
* Farmland — 90 ГБ;
* Satellite — 153 ГБ.

Итого примерно 807 ГБ. ([GitHub][12])

Для первой исследовательской версии A²CR это лишнее. Он пригодится позже, если появится отдельный claim о разных типах дымки: fog, cloud, environmental haze.

## Полный DIODE train

81 ГБ. Validation на 2,6 ГБ сначала полностью достаточно. ([diode-dataset.github.io][5])

## Полные RESIDE ITS/OTS

Они нужны главным образом для обучения. A²CR аналитический и не должен зависеть от огромной обучающей выборки.

---

# Рекомендуемая структура каталогов

```text
datasets/
├── archives/
│   ├── I-HAZE.zip
│   ├── O-HAZE.zip
│   ├── DENSE-HAZE.zip
│   ├── NH-HAZE.zip
│   └── diode_val.tar.gz
│
├── real_paired/
│   ├── i_haze/
│   │   ├── hazy/
│   │   └── clear/
│   ├── o_haze/
│   │   ├── hazy/
│   │   └── clear/
│   ├── dense_haze/
│   │   ├── hazy/
│   │   └── clear/
│   ├── nh_haze/
│   │   ├── hazy/
│   │   └── clear/
│   └── lmhaze/
│       ├── train/
│       └── test/
│
├── synthetic_external/
│   ├── reside_sots/
│   └── carla_haze/
│
├── rgbd_sources/
│   ├── diode_val/
│   ├── nyu_depth_v2/
│   └── middlebury/
│
├── synthetic_controlled/
│   ├── clear/
│   ├── depth/
│   ├── hazy/
│   ├── transmission/
│   └── metadata/
│
├── real_unpaired/
│   ├── reside_rtts/
│   ├── foggy_driving/
│   └── acdc/
│
└── manifests/
    ├── files_sha256.csv
    ├── dataset_manifest.csv
    └── experiment_splits.json
```

Исходные архивы лучше хранить неизменяемыми. После скачивания создать SHA-256 для каждого файла.

---

# Что ещё необходимо кроме фотографий

## 1. Зафиксированная гипотеза

Например:

> При ненадёжной transmission map и низком цветном SNR двухкоэффициентный airlight-aligned recovery снижает цветовой шум и clipping по сравнению со scalar (1/t), не ухудшая восстановление на участках с высокой уверенностью.

## 2. Baseline-методы

Минимально:

* canonical DCP;
* DCP с `tmin`;
* текущий SimpleDeHaze `chromaFloor`;
* RFEP;
* BCCR;
* Haze-Lines;
* A²CR без uncertainty;
* A²CR с uncertainty;
* A²CR с RGB-feasible projection.

## 3. Абляции

Нужно показать вклад каждого компонента:

```text
A²CR-base
+ linear RGB
+ transmission uncertainty
+ atmospheric-light uncertainty
+ noise model
+ feasible projection
+ spatial refinement
```

## 4. Метрики

Для парных наборов:

* PSNR;
* SSIM;
* CIEDE2000;
* LPIPS;
* hue error;
* chroma error.

Специально для A²CR:

* доля пикселей, которые до clamp вышли за `[0,1]`;
* величина clipping;
* усиление шума в плоских областях;
* ошибка (t) на synthetic данных;
* ошибка (A);
* ошибка (g_{\parallel}) и (g_{\perp});
* доля пикселей, где сработала feasible projection;
* время CPU/GPU;
* RAM/VRAM.

## 5. Зафиксированные разбиения

Нельзя использовать одну и ту же сцену одновременно для выбора параметров и финального теста.

Для DIODE деление должно быть **по сценам**, а не случайно по изображениям:

```text
development: 60%
validation: 20%
internal test: 20%
```

Финальные I-HAZE, O-HAZE, Dense-Haze, NH-HAZE и LMHaze-test остаются полностью закрытыми до фиксации метода.

---

# Конкретный порядок загрузки

1. [I-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire18/i-haze/)
2. [O-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/)
3. [Dense-Haze](https://data.vision.ee.ethz.ch/cvl/ntire19/dense-haze/)
4. [NH-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/)
5. [DIODE validation](http://diode-dataset.s3.amazonaws.com/val.tar.gz)
6. [RESIDE SOTS и RTTS](https://sites.google.com/view/reside-dehaze-datasets)
7. [CARLA-Haze](https://leoxthomas.github.io/CARLA-Haze/)
8. [LMHaze test](https://drive.google.com/file/d/1V-PwHbLfNgg1nF64bPK9HHUlc93bDSGs/view?usp=sharing)
9. [Foggy Driving](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/)

**Практический минимум для начала разработки:** четыре маленьких реальных набора + DIODE validation. Этого достаточно, чтобы построить собственный контролируемый генератор, доказать работу нового recovery operator и проверить его на настоящей дымке.

[1]: https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/ "https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/"
[2]: https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/ "https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/"
[3]: https://github.com/wangzrk/LMHaze "https://github.com/wangzrk/LMHaze"
[4]: https://sites.google.com/view/reside-dehaze-datasets "https://sites.google.com/view/reside-dehaze-datasets"
[5]: https://diode-dataset.org/ "https://diode-dataset.org/"
[6]: https://tensorflow.google.cn/datasets/catalog/nyu_depth_v2 "https://tensorflow.google.cn/datasets/catalog/nyu_depth_v2"
[7]: https://vision.middlebury.edu/stereo/data/ "https://vision.middlebury.edu/stereo/data/"
[8]: https://www.cvlibs.net/datasets/kitti/eval_depth_all.php "https://www.cvlibs.net/datasets/kitti/eval_depth_all.php"
[9]: https://leoxthomas.github.io/CARLA-Haze/ "https://leoxthomas.github.io/CARLA-Haze/"
[10]: https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/ "https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/"
[11]: https://acdc.vision.ee.ethz.ch/ "https://acdc.vision.ee.ethz.ch/"
[12]: https://github.com/tanvirnwu/HazeSpace2M "https://github.com/tanvirnwu/HazeSpace2M"
