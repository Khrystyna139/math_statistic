import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# ЛАБОРАТОРНА РОБОТА З МАТЕМАТИЧНОЇ СТАТИСТИКИ
# Варіант №8
# ============================================================

DATA = [
    7, 5, 6, 4, 5, 3, 8, 5, 6, 6, 5,
    8, 7, 5, 6, 6, 3, 6, 5, 5, 6, 7,
    5, 7, 6, 7, 5, 5, 6, 6, 5, 3, 4,
    5, 4, 5, 5, 5, 5, 6, 5, 3, 6, 6,
    5, 6, 5, 5, 6, 6, 6, 6, 8, 8, 7,
    3, 7, 9, 7, 3, 4, 5, 10, 3, 7, 6,
    5, 7, 4
]


def статистика(data):
    x = np.asarray(data, dtype=float)
    n = len(x)
    середнє = np.mean(x)
    дисперсія = np.mean((x - середнє) ** 2)

    counts = Counter(data)
    values = np.array(sorted(counts), dtype=float)
    frequencies = np.array([counts[int(v)] for v in values])
    relative = frequencies / n

    return {
        "n": n,
        "min": np.min(x),
        "max": np.max(x),
        "range": np.max(x) - np.min(x),
        "mean": середнє,
        "variance": дисперсія,
        "std": math.sqrt(дисперсія),
        "corrected_variance": np.var(x, ddof=1),
        "corrected_std": np.std(x, ddof=1),
        "cv": math.sqrt(дисперсія) / середнє * 100,
        "mode": values[np.argmax(frequencies)],
        "median": np.median(x),
        "q1": np.percentile(x, 25),
        "q3": np.percentile(x, 75),
        "values": values,
        "frequencies": frequencies,
        "relative": relative,
    }


def показати_дані():
    print("\nПочаткова вибірка:")
    print(DATA)
    print(f"Обсяг вибірки: n = {len(DATA)}")


def показати_статистичний_ряд(data=DATA):
    st = статистика(data)
    накопичена_частота = 0
    накопичена_відносна = 0

    print("\n" + "=" * 82)
    print("ВАРІАЦІЙНИЙ ТА СТАТИСТИЧНИЙ РЯД")
    print("=" * 82)
    print(f"{'xᵢ':>8}{'nᵢ':>10}{'wᵢ':>16}{'Nᵢ':>10}{'Wᵢ':>16}")
    print("-" * 82)

    for x, n_i, w_i in zip(
        st["values"], st["frequencies"], st["relative"]
    ):
        накопичена_частота += int(n_i)
        накопичена_відносна += w_i
        print(
            f"{int(x):>8}{int(n_i):>10}"
            f"{w_i:>16.6f}{накопичена_частота:>10}"
            f"{накопичена_відносна:>16.6f}"
        )

    print("-" * 82)
    print(f"{'Разом':>8}{st['n']:>10}{1.0:>16.6f}")


def показати_характеристики(data=DATA, назва="Початкова вибірка"):
    st = статистика(data)
    x = np.asarray(data, dtype=float)

    # Друга формула дисперсії:
    дисперсія_2 = np.mean(x ** 2) - np.mean(x) ** 2

    print("\n" + "=" * 82)
    print(f"ЧИСЛОВІ ХАРАКТЕРИСТИКИ: {назва.upper()}")
    print("=" * 82)
    print(f"Обсяг вибірки n                    = {st['n']}")
    print(f"Мінімум                             = {st['min']:.6f}")
    print(f"Максимум                            = {st['max']:.6f}")
    print(f"Розмах R                            = {st['range']:.6f}")
    print(f"Середнє арифметичне x̄              = {st['mean']:.6f}")
    print(f"Дисперсія D                         = {st['variance']:.6f}")
    print(f"СКВ σ                               = {st['std']:.6f}")
    print(f"Виправлена дисперсія s²             = {st['corrected_variance']:.6f}")
    print(f"Виправлене СКВ s                    = {st['corrected_std']:.6f}")
    print(f"Коефіцієнт варіації V               = {st['cv']:.2f}%")
    print(f"Мода Mo                             = {st['mode']:.6f}")
    print(f"Медіана Me                          = {st['median']:.6f}")
    print(f"Перший квартиль Q1                  = {st['q1']:.6f}")
    print(f"Третій квартиль Q3                  = {st['q3']:.6f}")
    print(
        f"Міжквартильний розмах IQR          = "
        f"{st['q3'] - st['q1']:.6f}"
    )

    print("\nПеревірка дисперсії другим способом:")
    print(f"D = (1/n)Σxᵢ² − x̄²                = {дисперсія_2:.6f}")


def побудувати_графіки(data=DATA):
    st = статистика(data)
    x = st["values"]
    частоти = st["frequencies"]
    відносні = st["relative"]
    накопичені = np.cumsum(відносні)

    # --------------------------------------------------------
    # 1. Полігон частот
    # --------------------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.plot(
        x, частоти,
        marker="o",
        linestyle="-",
        linewidth=1.8,
        markersize=6
    )
    plt.title("Полігон частот", fontsize=14)
    plt.xlabel("Варіанти ознаки xᵢ")
    plt.ylabel("Частоти nᵢ")
    plt.xticks(x)
    plt.yticks(range(0, int(max(частоти)) + 3, 2))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # --------------------------------------------------------
    # 2. Полігон відносних частот
    # --------------------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.plot(
        x, відносні,
        marker="o",
        linestyle="-",
        linewidth=1.8,
        markersize=6
    )
    plt.title("Полігон відносних частот", fontsize=14)
    plt.xlabel("Варіанти ознаки xᵢ")
    plt.ylabel("Відносні частоти wᵢ")
    plt.xticks(x)
    plt.yticks(np.arange(0, 0.36, 0.05))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # --------------------------------------------------------
    # 3. Емпірична функція розподілу
    # --------------------------------------------------------
    plt.figure(figsize=(9, 5))

    # Для F*(x) значення праворуч від кожного xᵢ дорівнює
    # накопиченій відносній частоті.
    x_step = np.r_[x[0] - 1, x, x[-1] + 1]
    y_step = np.r_[0, 0, накопичені]

    # Ступінчасте зображення емпіричної функції.
    plt.step(
        x_step,
        y_step,
        where="post",
        linewidth=1.8
    )

    # Точки-стрибки, характерні для емпіричної функції.
    plt.scatter(x, накопичені, s=35, zorder=3)

    plt.title("Емпірична функція розподілу", fontsize=14)
    plt.xlabel("x")
    plt.ylabel("F*(x)")
    plt.xticks(x)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1.05)
    plt.xlim(x[0] - 0.5, x[-1] + 0.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.show()


def перерахувати_до_100(data=DATA, m=100):
    """
    Перерахунок за квантильною функцією:
    L_j = (n+1)j/m.
    Якщо L_j не ціле, застосовується лінійна інтерполяція.
    """
    x = np.sort(np.asarray(data, dtype=float))
    n = len(x)
    результат = []

    for j in range(1, m + 1):
        L = (n + 1) * j / m

        if L <= 1:
            y = x[0]
        elif L >= n:
            y = x[-1]
        else:
            k = int(math.floor(L))
            alpha = L - k
            y = x[k - 1] + alpha * (x[k] - x[k - 1])

        результат.append(y)

    return np.array(результат)


def показати_перерахунок():
    нова = перерахувати_до_100()

    print("\n" + "=" * 82)
    print("ПЕРЕРАХУНОК ВИБІРКИ ДО m = 100")
    print("=" * 82)
    print("Формула: Lⱼ = (n+1)·j/m")
    print("Для нецілих позицій використовується лінійна інтерполяція.\n")

    for i in range(0, 100, 10):
        print(" ".join(f"{v:.1f}" for v in нова[i:i + 10]))

    показати_характеристики(нова, "Перерахована вибірка m=100")
    return нова


def порівняти_вибірки():
    стара = статистика(DATA)
    нова = статистика(перерахувати_до_100())

    print("\n" + "=" * 82)
    print("ПОРІВНЯННЯ ПОЧАТКОВОЇ ТА ПЕРЕРАХОВАНОЇ ВИБІРКИ")
    print("=" * 82)
    print(
        f"{'Характеристика':<30}"
        f"{'n=69':>14}{'m=100':>14}{'Зміна, %':>14}"
    )
    print("-" * 82)

    поля = [
        ("Середнє", "mean"),
        ("Дисперсія", "variance"),
        ("СКВ", "std"),
        ("Виправлена дисперсія", "corrected_variance"),
        ("Виправлене СКВ", "corrected_std"),
        ("Медіана", "median"),
        ("Мода", "mode"),
    ]

    for назва, поле in поля:
        a = стара[поле]
        b = нова[поле]
        зміна = abs(b - a) / abs(a) * 100 if a != 0 else 0
        print(f"{назва:<30}{a:>14.6f}{b:>14.6f}{зміна:>13.4f}%")


def повний_звіт():
    print("\n" + "#" * 82)
    print("ЛАБОРАТОРНА РОБОТА З МАТЕМАТИЧНОЇ СТАТИСТИКИ")
    print("ВАРІАНТ №8")
    print("#" * 82)
    print(
        "\nДосліджувана величина: кількість втрачених пакетів "
        "на 1000 переданих, шт."
    )

    показати_дані()
    показати_статистичний_ряд()
    показати_характеристики()
    показати_перерахунок()
    порівняти_вибірки()
    побудувати_графіки()


def меню():
    while True:
        print("\n" + "=" * 65)
        print("     МАТЕМАТИЧНА СТАТИСТИКА — ЛАБОРАТОРНА РОБОТА")
        print("                         ВАРІАНТ №8")
        print("=" * 65)
        print("1. Початкові дані")
        print("2. Варіаційний та статистичний ряд")
        print("3. Числові характеристики")
        print("4. Побудувати графіки")
        print("5. Перерахунок вибірки до m=100")
        print("6. Порівняння вибірок")
        print("7. Виконати всі завдання")
        print("0. Вихід")
        print("=" * 65)

        вибір = input("Оберіть пункт меню: ").strip()

        if вибір == "1":
            показати_дані()
        elif вибір == "2":
            показати_статистичний_ряд()
        elif вибір == "3":
            показати_характеристики()
        elif вибір == "4":
            побудувати_графіки()
        elif вибір == "5":
            показати_перерахунок()
        elif вибір == "6":
            порівняти_вибірки()
        elif вибір == "7":
            повний_звіт()
        elif вибір == "0":
            print("\nПрограму завершено.")
            break
        else:
            print("\nПомилка: введіть число від 0 до 7.")

        input("\nНатисніть Enter для повернення до меню...")


# Запуск меню тільки під час безпосереднього запуску цього файлу.
# У VS Code / PyCharm запускайте саме файл laboratorna_math_stat_8.py.
if __name__ == "__main__":
    меню()