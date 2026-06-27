# Adaptive Soft Dark Channel - мягкий и адаптивный DCP

Классический DCP берёт жёсткий минимум по каналам и окну:

$$D(x)=\min_{y\in\Omega_r(x)}\min_c I_c(y).$$

Это быстро и просто, но минимум слишком чувствителен к шуму, цветным артефактам, JPEG-блокам
и неверному размеру окна. Идея Adaptive Soft DCP - оставить физику DCP, но заменить жёсткий
минимум на устойчивую статистику и выбирать радиус окна по локальной структуре кадра.

> Статус: **реализовано** - `DCP - Adaptive Soft Dark Channel`
> ([`AdaptiveSoftDcpMethod.cs`](../../Methods/AdaptiveSoftDcpMethod.cs)): soft-min через
> box-фильтр числителя/знаменателя ($w=e^{-I_{min}/\tau}$).

## 1. Soft-min вместо min

Жёсткий минимум можно заменить soft-min:

$$\operatorname{softmin}_{\tau}(z_1,\dots,z_n)
=-\tau\log\sum_i \exp(-z_i/\tau).$$

На практике удобнее нормированный вариант:

$$D_\tau(x)=
\frac{\sum_{y,c} w_{y,c}\,I_c(y)}{\sum_{y,c} w_{y,c}},\qquad
w_{y,c}=\exp(-I_c(y)/\tau).$$

Малое $\tau$ приближает обычный min, большое $\tau$ делает оценку устойчивее.

## 2. Percentile / k-min dark channel

Ещё более простой вариант - брать не минимум, а нижний процентиль:

$$D_p(x)=\operatorname{percentile}_{p}\{I_c(y):y\in\Omega_r(x), c\in RGB\}.$$

Например, $p=1\%$ или $p=5\%$. Это убирает одиночные шумовые тёмные пиксели и делает карту
трансмиссии менее пятнистой.

## 3. Адаптивный радиус окна

Один и тот же `patch` плохо работает сразу в небе, листве и на тонких объектах. Радиус можно
выбирать по текстуре и градиенту:

$$r(x)=r_{\min} + (r_{\max}-r_{\min})\,
\exp(-\alpha\,\lVert\nabla Y(x)\rVert)\,
\exp(-\beta\,\operatorname{var}_{\Omega}(Y)).$$

- На гладких областях радиус больше: меньше шума и блочности.
- На краях и текстурах радиус меньше: меньше ореолов.

## Конвейер

```mermaid
flowchart LR
    I["I/A"] --> G["градиент/variance"]
    G --> R["адаптивный r(x)"]
    I --> SD["soft/percentile dark channel"]
    R --> SD
    SD --> T["t = 1 - omega * D"]
    T --> REF["Guided/WLS/MST"]
    REF --> J["Recover"]
```

## Псевдокод

```python
def adaptive_soft_dcp(I, A, r_min=3, r_max=15, tau=0.03, omega=0.5):
    N = I / A
    Y = gray(I)
    grad = abs_sobel(Y)
    tex = local_variance(Y, radius=5)

    r = r_min + (r_max - r_min) * exp(-4.0 * grad) * exp(-10.0 * tex)

    # вариант 1: soft-min в окне переменного радиуса
    D = adaptive_window_softmin(N, r, tau)

    # вариант 2 быстрее: несколько фиксированных dark channel и смешивание
    D_small = percentile_dark(N, radius=r_min, percentile=3)
    D_large = percentile_dark(N, radius=r_max, percentile=3)
    a = normalize(r - r_min, 0, r_max - r_min)
    D = (1 - a) * D_small + a * D_large

    t = clip(1 - omega * D, 0.05, 1.0)
    return edge_aware_refine(t, I)
```

## Быстрая реализация

Переменное окно дорого. Для проекта проще сделать дискретный набор радиусов:

1. Считать `D3`, `D7`, `D15` обычным/percentile min-filter.
2. Построить карту выбора радиуса по градиенту и variance.
3. Смешать карты плавными весами.

Это даёт почти тот же эффект, но остаётся GPU-friendly: несколько морфологических erosion
или быстрых min-filter проходов.

## Плюсы / минусы

| Плюсы | Минусы |
|---|---|
| Почти тот же DCP, но меньше шума и блочности | Percentile/soft-min дороже обычной эрозии |
| Меньше ореолов на тонких объектах | Нужны новые параметры: $\tau$, percentile, радиусы |
| Хороший первый кандидат для реализации | Может ослабить удаление плотного тумана |

## Связь с проектом

Заменяет [`DehazeCore.DarkChannel`](../../Methods/DehazeCore.cs) или только этап
`RawTransmission`. Остальной конвейер можно оставить прежним: `Atmospheric` -> `Recover`,
а для уточнения использовать уже существующие Guided/WLS/MST/Beltrami методы.
