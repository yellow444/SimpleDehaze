# LAF-TV/WLS - Low-Resolution Local Airlight Field

> Статус: **реализовано** - `LAF-TV/WLS (low-res airlight)`
> ([`LafTvMethod.cs`](../../Methods/LafTvMethod.cs)).

LAF-TV/WLS реализует направление LAF-TV из `TEMP.md`: вместо одной глобальной константы
атмосферного света используется медленно меняющееся поле `A(x)`. Это полезно для сцен, где
дымка или подсветка неоднородны: градиентное небо, контровой свет, городская дымка, glare.

## Модель

$$
I_c(x)=J_c(x)t(x)+A_c(x)(1-t(x)).
$$

В отличие от обычного DCP, `A` теперь не scalar BGR, а три одноканальных поля `A_c(x)`.

## Low-Res Airlight

На уменьшенной копии кадра `I_g` строится confidence:

$$
q(x)=D(x)^2(1-S(x))^{1.5}\exp(-2|\nabla Y(x)|).
$$

То есть кандидаты airlight должны быть светлыми в dark channel, малонасыщенными и гладкими.
Первичная оценка:

$$
A_c^0(x)=\frac{\operatorname{blur}(qI_{g,c})}{\operatorname{blur}(q)+\varepsilon}.
$$

Затем каждое поле `A_c` сглаживается `Refiners.Wls` на low-res сетке. Это WLS-прокси для
TV/WLS регуляризации из `TEMP.md`: не полноценный PCG/TV solver, но дешёвый edge-aware prior,
который не даёт `A(x)` переобучиться под текстуру.

## Upsample и Recovery

После WLS поле `A_g(x)` апсемплится на полное разрешение и дополнительно проходит
`Refiners.FastGuided` с исходным изображением как guide. Дальше:

$$
t(x)=1-\omega\,\operatorname{dark}\left(\frac{I(x)}{A(x)+\varepsilon}\right).
$$

Финальное восстановление использует chroma-safe локальную модель: яркость делится через
`t_min`, а хрома через `chromaFloor`, как в `DehazeCore.Recover`, но с `A_c(x)`.

## Параметры по умолчанию

| Параметр | Значение | Смысл |
|---|---:|---|
| `omega` | `0.95` | сила DCP |
| `patch` | `5` | радиус dark channel |
| `grid` | `12` | downsample-фактор для A(x) |
| `aRadius` | `6` | окно low-res оценки A(x) |
| `lambda` | `8` | WLS-гладкость поля A(x) |
| `iters` | `12` | итераций WLS |
| `min` | `0.10` | нижний порог яркостной трансмиссии |
| `chroma` | `0.35` | нижний порог хромы |
| `fast` | `4` | downsample fast guided filter |
| `color` | `1.25` | потолок усиления цветности |

## Абляции

- Global DCP: один `A`.
- `DCP - Local Airlight Field`: full-res blur-поле `A(x)`.
- `LAF-TV/WLS`: low-res `A_g`, WLS-регуляризация, guided upsample.

Такой набор напрямую проверяет тезис из `TEMP.md`: насколько ошибки дехейзинга идут от
плохой карты `t`, а насколько от слишком грубого предположения про глобальный атмосферный свет.
