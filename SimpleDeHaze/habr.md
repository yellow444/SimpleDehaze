﻿### Решение проблемы дымки на изображениях с использованием  .NET: Простой и эффективный подход

Дымка на изображениях может стать настоящей проблемой, и не всегда для ее удаления нужны сложные алгоритмы или нейронные сети. Я хочу продемонстрировать реализацию метода удаления дымки Robust Single Image Haze Removal Using Dark Channel Prior and Optimal Transmission Map and Adaptive Atmospheric Light (Удаление дымки с использованием метода предварительного темного канала, карты пропускания и не однородного света) в .NET.


#### Преимущества подхода:<a id="преимущества-подхода"></a>

1. **Простота и эффективность**: Метод, основанный на интуитивных принципах обработки изображений, легко воспринимается и прост в реализации. Использование только математических операций, и зная их параметры, в итоговом изображении никогда не будут присутствовать неизвестные артефакты или “фантомы”. Работа с изображением не будет выглядеть как копание в черном ящике с параметрами нейронной сети или библиотеки с закрытым алгоритмом.

2. **Отсутствие зависимости от обучающих данных**: Не требует сложного обучающего набора данных, что сокращает затраты и упрощает процесс.

3. **Высокая скорость работы**: Работает быстро и не требует значительных вычислительных ресурсов.

4. **Прозрачность и интерпретируемость**: Легко понимаемый метод обработки изображений, что упрощает объяснение и обоснование результатов.

5. **Универсальность**: Хорошо работает на различных типах изображений и условиях освещения без необходимости сложной перенастройки.


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
