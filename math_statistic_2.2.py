
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, chi2, norm


# ============================================================
# ЛАБОРАТОРНА РОБОТА № 2
# Варіант 8
# Статистичне оцінювання параметрів розподілу
# ============================================================

# Вхідні дані
left_bounds = np.array([60, 61, 62, 63, 64, 65, 66, 67])
right_bounds = np.array([61, 62, 63, 64, 65, 66, 67, 68])
frequencies = np.array([3, 6, 17, 22, 27, 16, 6, 2])

n = 99
h = 1.0
gamma = 0.99
alpha = 1 - gamma
df = n - 1

# Середини інтервалів
midpoints = (left_bounds + right_bounds) / 2

# Перевірка правильності вхідних даних
if np.sum(frequencies) != n:
    raise ValueError("Сума частот не дорівнює обсягу вибірки!")

# Основні статистичні характеристики
x_mean = np.sum(frequencies * midpoints) / n

sum_squared_deviations = np.sum(
    frequencies * (midpoints - x_mean) ** 2
)

d = sum_squared_deviations / n
s2 = sum_squared_deviations / (n - 1)
s = np.sqrt(s2)


# ============================================================
# ЗАВДАННЯ 1. Вхідний інтервальний ряд
# ============================================================

def task1():
    print("\nЗАВДАННЯ 1. ВХІДНИЙ ІНТЕРВАЛЬНИЙ РЯД")
    print("-" * 65)

    data = pd.DataFrame({
        "Інтервал, °C": [
            f"{left}–{right}"
            for left, right in zip(left_bounds, right_bounds)
        ],
        "Середина інтервалу": midpoints,
        "Частота": frequencies
    })

    print(data.to_string(index=False))
    print(f"\nКількість інтервалів: {len(frequencies)}")
    print(f"Сума частот: {np.sum(frequencies)}")
    print(f"Обсяг вибірки: {n}")
    print(f"Ширина інтервалу: {h} °C")

    print("\nПеревірка суми частот:")
    if np.sum(frequencies) == n:
        print("Сума частот відповідає обсягу вибірки.")


# ============================================================
# ЗАВДАННЯ 2. Гістограма та полігон частот
# ============================================================

def task2():
    print("\nЗАВДАННЯ 2. ГРАФІЧНЕ ЗОБРАЖЕННЯ РОЗПОДІЛУ")
    print("-" * 65)

    plt.figure(figsize=(10, 6))

    plt.bar(
        midpoints,
        frequencies,
        width=h,
        edgecolor="black",
        alpha=0.65,
        label="Гістограма частот"
    )

    plt.plot(
        midpoints,
        frequencies,
        marker="o",
        linewidth=2,
        label="Полігон частот"
    )

    plt.title("Розподіл температури в сушильній камері")
    plt.xlabel("Температура, °C")
    plt.ylabel("Частота")
    plt.xticks(midpoints)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    print("Відкривається вікно з гістограмою та полігоном частот.")
    print("За графіком можна візуально оцінити форму розподілу.")

    plt.show()


# ============================================================
# ЗАВДАННЯ 3. Точкові оцінки параметрів
# ============================================================

def task3():
    print("\nЗАВДАННЯ 3. ТОЧКОВІ ОЦІНКИ ПАРАМЕТРІВ")
    print("-" * 65)

    table = pd.DataFrame({
        "xᵢ": midpoints,
        "nᵢ": frequencies,
        "nᵢ · xᵢ": frequencies * midpoints,
        "(xᵢ - x̄)²": (midpoints - x_mean) ** 2,
        "nᵢ · (xᵢ - x̄)²":
            frequencies * (midpoints - x_mean) ** 2
    })

    print(table.round(4).to_string(index=False))

    print("\nПідсумкові результати:")
    print(f"Обсяг вибірки n = {n}")
    print(f"Середнє значення x̄ = {x_mean:.6f}")
    print(f"Зміщена дисперсія d = {d:.6f}")
    print(f"Незміщена дисперсія s² = {s2:.6f}")
    print(f"Стандартне відхилення s = {s:.6f}")

    print(f"\nΣ nᵢxᵢ = {np.sum(frequencies * midpoints):.4f}")
    print(
        "Σ nᵢ(xᵢ - x̄)² = "
        f"{sum_squared_deviations:.6f}"
    )


# ============================================================
# ЗАВДАННЯ 4. Довірчий інтервал для математичного сподівання
# ============================================================

def task4():
    print("\nЗАВДАННЯ 4. ДОВІРЧИЙ ІНТЕРВАЛ ДЛЯ СЕРЕДНЬОГО")
    print("-" * 65)

    t_critical = t.ppf(1 - alpha / 2, df)
    delta = t_critical * s / np.sqrt(n)

    lower = x_mean - delta
    upper = x_mean + delta

    relative_precision = delta / abs(x_mean) * 100

    print(f"Рівень надійності γ = {gamma:.2f}")
    print(f"Рівень значущості α = {alpha:.2f}")
    print(f"Ступені вільності = {df}")
    print(f"Критичне значення Стьюдента = {t_critical:.6f}")
    print(f"Точність оцінки δ = {delta:.6f}")

    print(
        f"\nДовірчий інтервал для m: "
        f"[{lower:.4f}; {upper:.4f}] °C"
    )

    print(f"Відносна точність ε = {relative_precision:.4f}%")


# ============================================================
# ЗАВДАННЯ 5. Довірчі інтервали для дисперсії та відхилення
# ============================================================

def task5():
    print("\nЗАВДАННЯ 5. ДОВІРЧІ ІНТЕРВАЛИ ДЛЯ σ² ТА σ")
    print("-" * 65)

    chi2_lower = chi2.ppf(alpha / 2, df)
    chi2_upper = chi2.ppf(1 - alpha / 2, df)

    variance_lower = sum_squared_deviations / chi2_upper
    variance_upper = sum_squared_deviations / chi2_lower

    std_lower = np.sqrt(variance_lower)
    std_upper = np.sqrt(variance_upper)

    print(f"Критичне значення χ²(α/2) = {chi2_lower:.6f}")
    print(f"Критичне значення χ²(1 - α/2) = {chi2_upper:.6f}")

    print(
        f"\nДовірчий інтервал для дисперсії σ²: "
        f"[{variance_lower:.4f}; {variance_upper:.4f}] °C²"
    )

    print(
        f"Довірчий інтервал для стандартного відхилення σ: "
        f"[{std_lower:.4f}; {std_upper:.4f}] °C"
    )


# ============================================================
# ЗАВДАННЯ 6. Точність і ширина довірчих інтервалів
# ============================================================

def task6():
    print("\nЗАВДАННЯ 6. ТОЧНІСТЬ ТА ШИРИНА ІНТЕРВАЛІВ")
    print("-" * 65)

    t_critical = t.ppf(1 - alpha / 2, df)
    delta = t_critical * s / np.sqrt(n)

    mean_lower = x_mean - delta
    mean_upper = x_mean + delta

    chi2_lower = chi2.ppf(alpha / 2, df)
    chi2_upper = chi2.ppf(1 - alpha / 2, df)

    variance_lower = sum_squared_deviations / chi2_upper
    variance_upper = sum_squared_deviations / chi2_lower

    std_lower = np.sqrt(variance_lower)
    std_upper = np.sqrt(variance_upper)

    print(f"Точність оцінки середнього δ = {delta:.6f}")
    print(
        f"Відносна точність ε = "
        f"{delta / abs(x_mean) * 100:.4f}%"
    )

    print(
        f"\nШирина інтервалу для m = "
        f"{mean_upper - mean_lower:.6f} °C"
    )

    print(
        f"Ширина інтервалу для σ² = "
        f"{variance_upper - variance_lower:.6f} °C²"
    )

    print(
        f"Ширина інтервалу для σ = "
        f"{std_upper - std_lower:.6f} °C"
    )

    print(
        "\nІнтервал для дисперсії несиметричний, оскільки "
        "розподіл хі-квадрат не є симетричним."
    )


# ============================================================
# ЗАВДАННЯ 7. Порівняння різних рівнів надійності
# ============================================================

def task7():
    print("\nЗАВДАННЯ 7. ПОРІВНЯННЯ РІВНІВ НАДІЙНОСТІ")
    print("-" * 65)

    confidence_levels = [0.90, 0.95, 0.99]
    rows = []

    for current_gamma in confidence_levels:
        current_alpha = 1 - current_gamma

        # Інтервал за розподілом Стьюдента
        t_critical = t.ppf(1 - current_alpha / 2, df)
        delta_t = t_critical * s / np.sqrt(n)

        lower_t = x_mean - delta_t
        upper_t = x_mean + delta_t

        # Інтервал за нормальним наближенням
        z_critical = norm.ppf(1 - current_alpha / 2)
        delta_z = z_critical * s / np.sqrt(n)

        lower_z = x_mean - delta_z
        upper_z = x_mean + delta_z

        rows.append({
            "γ": current_gamma,
            "t критичне": t_critical,
            "Інтервал Стьюдента":
                f"[{lower_t:.4f}; {upper_t:.4f}]",
            "δ Стьюдента": delta_t,
            "z критичне": z_critical,
            "Нормальний інтервал":
                f"[{lower_z:.4f}; {upper_z:.4f}]",
            "δ нормального": delta_z
        })

    table = pd.DataFrame(rows)
    print(table.round(5).to_string(index=False))

    print(
        "\nЗа збільшення рівня надійності інтервал стає ширшим. "
        "Нормальний розподіл використовується тут для порівняння "
        "зі значеннями розподілу Стьюдента."
    )


# ============================================================
# ЗАВДАННЯ 8. Обсяг вибірки для зменшення точності вдвічі
# ============================================================

def task8():
    print("\nЗАВДАННЯ 8. НЕОБХІДНИЙ ОБСЯГ ВИБІРКИ")
    print("-" * 65)

    new_n = 4 * n

    print(f"Початковий обсяг вибірки n = {n}")
    print("Для зменшення точності оцінки вдвічі:")
    print("δ пропорційна 1 / √n, тому обсяг вибірки збільшується в 4 рази.")
    print(f"Новий обсяг вибірки n' = 4 · {n} = {new_n}")
    print(f"Додатково потрібно {new_n - n} спостережень.")


# ============================================================
# ЗАВДАННЯ 9. Висновки
# ============================================================

def task9():
    print("\nЗАВДАННЯ 9. ВИСНОВКИ")
    print("-" * 65)

    t_critical = t.ppf(1 - alpha / 2, df)
    delta = t_critical * s / np.sqrt(n)

    mean_lower = x_mean - delta
    mean_upper = x_mean + delta

    chi2_lower = chi2.ppf(alpha / 2, df)
    chi2_upper = chi2.ppf(1 - alpha / 2, df)

    variance_lower = sum_squared_deviations / chi2_upper
    variance_upper = sum_squared_deviations / chi2_lower

    print(f"Обсяг вибірки: {n}")
    print(f"Середня температура: {x_mean:.4f} °C")
    print(f"Незміщена оцінка дисперсії: {s2:.4f} °C²")
    print(f"Стандартне відхилення: {s:.4f} °C")

    print(
        f"\nДовірчий інтервал для математичного сподівання "
        f"при γ = {gamma:.2f}: "
        f"[{mean_lower:.4f}; {mean_upper:.4f}] °C"
    )

    print(
        f"Довірчий інтервал для дисперсії: "
        f"[{variance_lower:.4f}; {variance_upper:.4f}] °C²"
    )

    print(
        "\nГістограма та полігон частот дають змогу візуально "
        "оцінити форму розподілу температури."
    )

    print(
        "Для зменшення точності оцінки середнього вдвічі "
        f"за наближеним правилом потрібно {4 * n} спостережень."
    )


# ============================================================
# ВИКОНАННЯ ВСІХ ЗАВДАНЬ
# ============================================================

def all_tasks():
    task1()
    task2()
    task3()
    task4()
    task5()
    task6()
    task7()
    task8()
    task9()


# ============================================================
# ГОЛОВНЕ МЕНЮ
# ============================================================

def show_menu():
    print("\n" + "=" * 65)
    print("ЛАБОРАТОРНА РОБОТА № 2 — ВАРІАНТ 8")
    print("=" * 65)
    print("Оберіть завдання:")
    print("1  — Вхідний інтервальний ряд")
    print("2  — Гістограма та полігон частот")
    print("3  — Точкові оцінки параметрів")
    print("4  — Довірчий інтервал для математичного сподівання")
    print("5  — Довірчі інтервали для дисперсії та відхилення")
    print("6  — Точність і ширина довірчих інтервалів")
    print("7  — Порівняння рівнів надійності")
    print("8  — Необхідний обсяг вибірки")
    print("9  — Висновки")
    print("10 — Виконати всі завдання")
    print("0  — Вихід")
    print("=" * 65)


def main():
    while True:
        show_menu()
        choice = input("Введіть номер завдання: ").strip()

        if choice == "1":
            task1()
        elif choice == "2":
            task2()
        elif choice == "3":
            task3()
        elif choice == "4":
            task4()
        elif choice == "5":
            task5()
        elif choice == "6":
            task6()
        elif choice == "7":
            task7()
        elif choice == "8":
            task8()
        elif choice == "9":
            task9()
        elif choice == "10":
            all_tasks()
        elif choice == "0":
            print("\nПрограму завершено.")
            break
        else:
            print("\nПомилка! Введіть номер від 0 до 10.")

        input("\nНатисніть Enter, щоб повернутися до меню...")


# Запуск програми
if __name__ == "__main__":
    main()