# Проверка литературы для A²CR и CAR (30 июля 2026)

Цель проверки - не доказать приоритет поиском по ключевым словам, а ограничить публикационные
формулировки. Поиск выполнен по первичным страницам авторов, arXiv и CVF Open Access. Отсутствие
точного совпадения в найденных работах **не является доказательством мирового первенства**.

## Что именно искалось

Запросы комбинировали `single image dehazing` с `uncertainty`, `analytic risk`, `dual gain`,
`airlight aligned`, `RGB gamut/feasible constraint`, `local atmospheric light`, `color constraint`,
`primal dual` и `total variation`. Отдельно проверены работы 2009-2026 годов о DCP, boundary
constraint, haze-lines, локальном airlight, uncertainty-aware dehazing и современных обучаемых
методах. Дата последнего поиска: 30 июля 2026 года.

## Ближайшие известные идеи

| Работа | Что уже известно | Чем отличается текущая реализация |
|---|---|---|
| [Dark Channel Prior](https://people.csail.mit.edu/kaiming/cvpr09/index.html), He et al., CVPR 2009 / TPAMI 2011 | Физическая модель, prior для `t`, скалярная инверсия `1/t` | A²CR меняет оператор восстановления после оценки `t`: два gain вдоль/поперёк airlight |
| [Boundary Constraint and Contextual Regularization](https://openaccess.thecvf.com/content_iccv_2013/html/Meng_Efficient_Image_Dehazing_2013_ICCV_paper.html), Meng et al., ICCV 2013 | RGB-границы используются для ограничения transmission | A²CR строит точный 2D-многоугольник допустимых **двух gain** на каждом пикселе |
| [Color Attenuation Prior](https://ueaeprints.uea.ac.uk/id/eprint/62621/), Zhu et al., TIP 2015 | Глубина из HSV saturation/value; обученная линейная модель | CAR использует CAP как оценку глубины, но восстанавливает radiance в linear RGB и отдельно переносит низкочастотную chroma |
| [Non-Local Image Dehazing](https://openaccess.thecvf.com/content_cvpr_2016/html/Berman_Non-Local_Image_Dehazing_CVPR_2016_paper.html), Berman et al., CVPR 2016 | Haze-lines в RGB, детерминированное восстановление без обучения | В A²CR упрощённая haze-line карта входит только в ensemble uncertainty; это не переизобретение Haze-Lines |
| [Airlight Field Estimation](https://arxiv.org/abs/1805.02142), Zhang et al., 2018 | Пространственно меняющееся поле airlight и совместная MAP-оценка | CAR специально отделяет локальную яркость `A(x)` от глобального chromatic anchor, чтобы поле не поглощало цвет большой поверхности |
| [Color-Constrained Dehazing Model](https://openaccess.thecvf.com/content_CVPRW_2020/html/w51/Zhang_Color-Constrained_Dehazing_Model_CVPRW_2020_paper.html), Zhang et al., CVPRW 2020 | Локальный airlight, TV и статистическое ограничение натуральных RGB-цветов | Это самая близкая работа к CAR; CAR не учит миллион-изображений color prior, а использует centered-RGB residual, coarse chroma и confidence от gamut projection |
| [Bounded Channel Difference Prior](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Zhao_Single_Image_Dehazing_Using_Bounded_Channel_Difference_Prior_CVPRW_2021_paper.html), Zhao, CVPRW 2021 | Ограничение разности каналов для устойчивого классического dehazing | Не содержит airlight-aligned двухкомпонентного риска A²CR |
| [Semi-UFormer](https://arxiv.org/abs/2210.16057), Tong et al., 2022 | Обучаемая pixel uncertainty как guidance teacher-student сети | Uncertainty A²CR аналитически выводится из bootstrap `A`, dispersion `D=-ln(t)` и noise; обучение отсутствует |
| [BPUL](https://arxiv.org/abs/2607.11623), Wei, 2026 | Обучаемая perturbation-induced uncertainty для real-world dehazing | Недавняя обучаемая framework-работа; не аналитический constrained recovery operator |
| [Condat primal-dual splitting](https://lcondat.github.io/publis/Condat-optim-JOTA-2013.pdf), 2013 | Общий алгоритм для smooth + proximable + linear-composite convex задач | Используется как solver, а не заявляется как вклад; вкладом-кандидатом является конкретная A²CR objective и её RGB-feasible prox |

Современные методы, например [CoA](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_CoA_Towards_Real_Image_Dehazing_via_Compression-and-Adaptation_CVPR_2025_paper.html),
[Learning Hazing to Dehazing](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Learning_Hazing_to_Dehazing_Towards_Realistic_Haze_Generation_for_Real-World_CVPR_2025_paper.html)
и [DehazeSB](https://openaccess.thecvf.com/content/ICCV2025/html/Lan_When_Schrodinger_Bridge_Meets_Real-World_Image_Dehazing_with_Unpaired_Training_ICCV_2025_paper.html),
показывают актуальный learning/diffusion контекст. A²CR не претендует на их абсолютное качество;
его ниша - обучение не требуется, формулы и ограничения проверяемы, CPU-исполнение детерминировано.

## Допустимая формулировка новизны

По найденной литературе не обнаружен прямой аналог, который одновременно объединяет:

1. разложение residual на направления вдоль и поперёк вектора airlight;
2. два аналитических gain из квадратичного риска с uncertainty `t`, `A` и noise;
3. точное per-pixel допустимое множество этих gain, полученное из шести RGB-неравенств;
4. совместную edge-aware TV-задачу с проекцией на этот многоугольник;
5. training-free восстановление.

Поэтому в статье используется фраза **"we introduce an analytically motivated combination"**,
а не `first`, `novel in the world` или `state of the art`. CAR описывается как отдельная
проверяемая гипотеза/инженерный case study, потому что его ближайшее родство с color-constrained
и local-airlight моделями существенно.

## Метрики и данные

- [SSIM](https://ece.uwaterloo.ca/~z70wang/publications/ssim.html): Wang et al., TIP 2004.
- [CIEDE2000 implementation notes and test data](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/): Sharma et al., 2005.
- [LPIPS](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf): Zhang et al., CVPR 2018.
- [O-HAZE](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Ancuti_O-HAZE_A_Dehazing_CVPR_2018_paper.html),
  [I-HAZE](https://arxiv.org/abs/1804.05091), [Dense-Haze](https://arxiv.org/abs/1904.02904),
  [NH-HAZE](https://arxiv.org/abs/2005.03560) и [DIODE](https://diode-dataset.org/).

Практический минимум данных локально собран. Real-paired наборы уже участвовали в разработке,
поэтому результаты на них являются fixed regression evaluation, но не слепым внешним тестом.
