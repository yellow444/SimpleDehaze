### Решение проблемы дымки на изображениях с использованием  .NET: Простой и эффективный подход

Дымка на изображениях может стать настоящей проблемой, и не всегда для ее удаления нужны сложные алгоритмы или нейронные сети. Я хочу продемонстрировать реализацию метода удаления дымки Robust Single Image Haze Removal Using Dark Channel Prior and Optimal Transmission Map and Adaptive Atmospheric Light (Удаление дымки с использованием метода предварительного темного канала, карты пропускания и не однородного света) в .NET.


#### Преимущества подхода:<a id="преимущества-подхода"></a>

1. **Простота и эффективность**: Метод, основанный на интуитивных принципах обработки изображений, легко воспринимается и прост в реализации. Использование только математических операций, и зная их параметры, в итоговом изображении никогда не будут присутствовать неизвестные артефакты или “фантомы”. Работа с изображением не будет выглядеть как копание в черном ящике с параметрами нейронной сети или библиотеки с закрытым алгоритмом.

2. **Отсутствие зависимости от обучающих данных**: Не требует сложного обучающего набора данных, что сокращает затраты и упрощает процесс.

3. **Высокая скорость работы**: Работает быстро и не требует значительных вычислительных ресурсов.

4. **Прозрачность и интерпретируемость**: Легко понимаемый метод обработки изображений, что упрощает объяснение и обоснование результатов.

5. **Универсальность**: Хорошо работает на различных типах изображений и условиях освещения без необходимости сложной перенастройки.


#### Используемые инструменты:<a id="используемые-инструменты"></a>

Для решения задачи удаления дымки мы используем библиотеку EmguCV, обертку для OpenCV в .NET. Этот инструмент обеспечивает удобный доступ к широкому спектру функций обработки изображений и видео, и матриц вообще. Причем синтаксис для работы с CPU и GPU примерно одинаковый new Mat() или new GpuMat(). Но есть отличие в вызове методов, которое унаследовано из OpenCV . И при работе с GpuMat требуется более тщательно следить за сборкой мусора, или реализовать свой интерфейс с GC, или придется постоянно использовать using. На github есть issue по очистке памяти GpuMat , но, пока оно не закрыто.

![](https://lh7-us.googleusercontent.com/u8QdlDRVEWBnKRd4rX97s3qHZJXF8TTNP8Y6pyZaiL0MF4V4JGR8uq2Zj3CWlgJhu8YzI-CjXLrzmzRugFBZGTKp_CCMsFXgAmkOzUX8MqBayjSKDd-fVxepolbLm6AJUDUAUGhXtYHmztAS4jvxWoA)

Схема атмосферного света была любезно предоставлена [mirasnowfox](https://mirasnowfox.ru)

Основные компоненты этого метода включают:

1. **Оценка атмосферного света**: 

используя квадратное разложение 

![](https://lh7-us.googleusercontent.com/MBGFGvPvr7e5AGLxCjxBDtTjuMyTveoWfQyYqAGirOcxE_woUN_Sq8Pb0xviVYpUQVcmPd9oQV1izoTISRx_oIVjdzOdbQd3OGe4MqazlQlTa7IU-b1hf7R6H4IKiUZvfPLteTNLFJnnhJCJSevvuQ0)

выберем область с наибольшей яркостью

![](https://lh7-us.googleusercontent.com/2fdeI_M6Bt4kl4gdtBQYhk2WGaqRyY-ejAWzKF9xJg4f2CC9RJGn6t3aQxZ3uq8g9a7X0eZTbROkBxBcEHVlu9Je3h2LeGGnaavVHRlxKof7-ZAOixZZ0RCBDG2cN2XoP-4sV9bCAF6Xy5hiU2wpvNQ)

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

![](https://lh7-us.googleusercontent.com/f8E_Kt3UeW3YenIuOmFHO7sKls0D32EbGVB-Yn6RpXJbkH3RJ9UQ4rKSypwhkiRj_p9_N-FF_PitvjwCN24X6gTFqnTxLt4yYc1HhyBbesmFh6O3BwCWmhaZYFDurdUfSIdDO6oLiklnyMpJUg74_mI)

по формуле $t(x,y)=e^{-\beta*d(x,y)}$ где  коэффициент ослабления атмосферы, которая оптимально отражает степень проникновения света через туман в каждой точке изображения. Хотя есть более простая альтернатива $t(x,y)=1-{\omega}*d(x,y)$ где $\omega$ количество дымки для удаления

3. **Уточнение карты трансмиссии**: 

к полученной карте трансмиссии применим Guided Filter, для 

![](https://lh7-us.googleusercontent.com/EpXifzPD7bwZ1OH_tE0-A-dwtSIQM_bWQ8BkLFYyeiVKceOkP2BypEAGa9V9IJdU8zekfKtCyCGZdyR_4T18NPy7iZx22nUPJ90K_Gu7RkmSt9m4uYOqmusaFsQV1-k4-jhKZYA5amIuEY_dHTEI-vA)

для смягчения краев у ярких мест изображения. Это могут быть источники света, места с сильным эффектом дымки, различные отражающие поверхности. Для GPU версии я использовал работу Kaiming He (<kahe@microsoft.com>) реализация в MATLAB http\://research.microsoft.com/en-us/um/people/kahe/eccv10/guided-filter-code-v1.rar

4. **Восстановление изображения**: 

производится по формуле: $J(x,y)=\frac{I(x,y)-A_c}{max(t(x,y),t_{min})}+A_c$ где I(x,y) - значение туманного пикселя, J(x,y) - значение безтуманного пикселя.

![](https://lh7-us.googleusercontent.com/-2Jv8lAiLDV2yi6_zcllph45fUUCE76UtU9Sif3_g_yMMp8zSYTP-zCLqUVVDMBC6Ob5L-QfNWLjqMnEzYnxLlibwpFhJ0DOdiL1bQm1sJ3Qq8tXqawGy8r4X4ESFR0J_g_YH-x7khVrunfSt6EvQqg)

Для сравнение изображение перед обработкой

![](https://lh7-us.googleusercontent.com/osOIAnr31i2tAHcI95KeerN0GkOiYW4DJR20HRhsK1vGqarB3jvFU-vXSpkFP3y9XHIApSBCdFwE_IGcdXQszR2sJtVPZhc0Ahqf_NifBc8JQPIN3cRYRiNF6tyYwSLX2hRl2WdBeO7YXe1hwFMV9OE)

Применение такого простого метода может значительно улучшить качество изображений и облегчить их последующий анализ и обработку.

Ссылка на проект [github](https://github.com/yellow444/SimpleDehaze/) 
