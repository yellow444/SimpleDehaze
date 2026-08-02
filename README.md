### Решение проблемы дымки на изображениях с использованием  .NET: Простой и эффективный подход

Дымка на изображениях может стать настоящей проблемой, и не всегда для ее удаления нужны сложные алгоритмы или нейронные сети. Я хочу продемонстрировать реализацию метода удаления дымки Robust Single Image Haze Removal Using Dark Channel Prior and Optimal Transmission Map and Adaptive Atmospheric Light (Удаление дымки с использованием метода предварительного темного канала, карты пропускания и не однородного света) в .NET.

Текущий центральный эксперимент — [A²CR-Dehaze](SimpleDeHaze/docs/methods/a2cr-dehaze.md):
airlight-aligned оператор восстановления с двумя uncertainty-aware gains и точной RGB-feasible
проекцией. Это проверяемый кандидат на исследовательский вклад, а не заявленная мировая новизна.
Для цвета теперь есть три уровня эксперимента. [HSV²CR](SimpleDeHaze/docs/methods/hsv-a2cr.md)
и [C³R-HSV](SimpleDeHaze/docs/methods/c3r-hsv.md) сохранены как отрицательные абляции.
[HCV-A²CR](SimpleDeHaze/docs/methods/hcv-a2cr-utaw.md) использует точную airlight-normalized
репараметризацию linear-RGB atmospheric model, два gain и RGB-feasible polygon. На frozen test
HCV/fusion немного подняли SSIM, но ухудшили PSNR, DE00 и clipping; основной A²CR они не заменяют.

Для визуально сильной perceptual-ветви реализованы
[Transmission-aware Laplacian, HSV Edge и stationary UTAW](SimpleDeHaze/docs/methods/transmission-aware-multiscale.md).
Validation-selected UTAW проверен на 91 test-паре O-/I-/Dense-/NH-HAZE: SSIM лучше старого
варианта на 91/91 и Edge на 86/91, однако flat-noise хуже Edge на 89/91. Hybrid CUDA численно
эквивалентен CPU, но end-to-end на RTX 3080 на 1.9% медленнее. Это structural-quality кандидат,
а не speed/SOTA claim. [Полный NEW3 audit](SimpleDeHaze/docs/research/hcv-a2cr-utaw-study-2026-08.md).
Данные, DIODE scene split, 22 500 controlled recipes, полный прогон 112 500 строк, настоящий LPIPS
на 364 real-paired результатах и честные отрицательные выводы зафиксированы в
[A²CR data protocol](SimpleDeHaze/docs/research/a2cr-data-protocol.md).

Публикационные материалы: [arXiv source](paper/a2cr-dehaze/README.md),
[проверенный PDF](output/pdf/a2cr-recovery-preprint.pdf),
[новая статья для Habr](SimpleDeHaze/docs/articles/habr-a2cr.md),
[CAR и сцена №08](SimpleDeHaze/docs/methods/car-dehaze.md).

Публикационное решение после NEW3: пока сохраняется одна основная статья A²CR. HCV добавляется
как математическая/отрицательная абляция, UTAW — как ongoing branch; отдельные статьи появятся
только после blind study и устранения noise/clipping trade-off.


#### Преимущества подхода:<a id="преимущества-подхода"></a>

1. **Прослеживаемость artefact'ов**: только математические операции, поэтому происхождение любого
   артефакта прослеживается до конкретной формулы — prior, уточнение, восстановление или постобработка.
   Это **не** значит, что артефактов нет: детерминированный алгоритм так же даёт ореолы, блочность
   от морфологии, клиппинг, перенасыщение, цветовые сдвиги и усиление шума. Ожидаемые режимы отказа
   перечислены в [NOVELTY.md](NOVELTY.md) и в документации методов.

2. **Отсутствие зависимости от обучающих данных**: не требует обучающего набора. Это уменьшает
   один источник доменного сдвига, но prior/model mismatch между типами сцен всё равно остаётся.

3. **Скорость**: сопоставима с другими классическими priors; конкретные числа зависят от кадра,
   железа и метода — см. колонки `ms`/`ms_per_mp` в CSV бенчмарка, а не общие утверждения.

4. **Прозрачность и интерпретируемость**: у каждой стадии есть формула, параметры видимы и
   поддаются подбору.

5. **Границы применимости**: приоры на основе тёмного канала систематически ошибаются на небе,
   белых объектах, снегу и бликах; неоднородная дымка нарушает предположение о постоянном
   атмосферном свете. Универсальности здесь нет — есть набор методов с разными зонами силы.


#### Используемые инструменты:<a id="используемые-инструменты"></a>

Для решения задачи удаления дымки мы используем библиотеку EmguCV, обертку для OpenCV в .NET. Этот инструмент обеспечивает удобный доступ к широкому спектру функций обработки изображений и видео, и матриц вообще. Причем синтаксис для работы с CPU и GPU примерно одинаковый new Mat() или new GpuMat(). Но есть отличие в вызове методов, которое унаследовано из OpenCV . И при работе с GpuMat требуется более тщательно следить за сборкой мусора, или реализовать свой интерфейс с GC, или придется постоянно использовать using. На github есть issue по очистке памяти GpuMat , но, пока оно не закрыто.

![](https://raw.githubusercontent.com/yellow444/SimpleDehaze/master/SimpleDeHaze/docs/light.jpg)

Схема атмосферного света была любезно предоставлена [mirasnowfox](https://mirasnowfox.ru)

Основные компоненты этого метода включают:

1. **Оценка атмосферного света**: 

используя квадратное разложение 

![](https://raw.githubusercontent.com/yellow444/SimpleDehaze/master/SimpleDeHaze/docs/image1.jpg)

выберем область с наибольшей яркостью

![](https://raw.githubusercontent.com/yellow444/SimpleDehaze/master/SimpleDeHaze/docs/image2.jpg)

так мы скорее всего избежим посторонние источники света, например, фары машин, и ускорим последующую сортировку. Далее для полученного участка находим его темный канал: простой, но эффективный способ оценить информацию о глубине сцены. Пример реализации 

~~~

private Mat ComputeDarkChannelPatch(Image<Bgr, float> srcImage, int patch)
        {
            var bgrChannels = srcImage.Clone().Mat.Split();
            var darkChannel = new Mat();
            CvInvoke.Min(bgrChannels[0], bgrChannels[1], darkChannel);
            CvInvoke.Min(darkChannel, bgrChannels[2], darkChannel);
            CvInvoke.Erode(darkChannel, darkChannel, null, new Point(-1, -1), patch, BorderType.Reflect101, default);
            return darkChannel;
        }
~~~

        Формула $d(x,y)=min(R(x,y),B(x,y),G(x,y))$ где $R(x,y)$ , $B(x,y)$ и $G(x,y)$ представляют интенсивность красного, зеленого и синего каналов для каждого пикселя соответственно. Сортируем и выбираем некоторый процент наиболее ярких пикселей, затем вычисляем для них среднее значения Ac по каждому каналу RGB.

 

2. **Построение оптимальной карты трансмиссии**: 

строим карту трансмиссии 

![](https://raw.githubusercontent.com/yellow444/SimpleDehaze/master/SimpleDeHaze/docs/image3.jpg)

по формуле $t(x,y)=e^{-\beta*d(x,y)}$ где  коэффициент ослабления атмосферы, которая оптимально отражает степень проникновения света через туман в каждой точке изображения. Хотя есть более простая альтернатива $t(x,y)=1-{\omega}*d(x,y)$ где $\omega$ количество дымки для удаления

3. **Уточнение карты трансмиссии**: 

к полученной карте трансмиссии применим Guided Filter, для 

![](https://raw.githubusercontent.com/yellow444/SimpleDehaze/master/SimpleDeHaze/docs/image4.jpg)

для смягчения краев у ярких мест изображения. Это могут быть источники света, места с сильным эффектом дымки, различные отражающие поверхности. Для GPU версии я использовал работу Kaiming He (<kahe@microsoft.com>) реализация в MATLAB http\://research.microsoft.com/en-us/um/people/kahe/eccv10/guided-filter-code-v1.rar

4. **Восстановление изображения**: 

производится по формуле: $J(x,y)=\frac{I(x,y)-A_c}{max(t(x,y),t_{min})}+A_c$ где I(x,y) - значение туманного пикселя, J(x,y) - значение безтуманного пикселя.

![](https://raw.githubusercontent.com/yellow444/SimpleDehaze/master/SimpleDeHaze/docs/01_outdoor_hazy_Cpu.jpg)

Для сравнение изображение перед обработкой

![](https://raw.githubusercontent.com/yellow444/SimpleDehaze/master/SimpleDeHaze/docs/01_outdoor_hazy.jpg)

Применение такого простого метода может значительно улучшить качество изображений и облегчить их последующий анализ и обработку.

Ссылка на проект [github](https://github.com/yellow444/SimpleDehaze/) 
