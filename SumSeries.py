import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# Определение функции
def f(x):
    return x + 2 * np.sin(3 * x)


# Параметры
L = np.pi  # Интервал [-π, π]
N = 30  # Количество гармоник
RESOLUTION = 1000  # Количество точек для графика


# Вычисление коэффициентов Фурье
def calculate_coefficients():
    # a0
    a0, _ = quad(lambda x: f(x), -L, L)
    a0 /= (2 * L)

    # an и bn
    an = np.zeros(N)
    bn = np.zeros(N)

    for n in range(1, N + 1):
        # an (интеграл для косинусов)
        integrand_cos = lambda x: f(x) * np.cos(n * x)
        an[n - 1], _ = quad(integrand_cos, -L, L)
        an[n - 1] /= L

        # bn (интеграл для синусов)
        integrand_sin = lambda x: f(x) * np.sin(n * x)
        bn[n - 1], _ = quad(integrand_sin, -L, L)
        bn[n - 1] /= L

    return a0, an, bn


a0, an, bn = calculate_coefficients()

print("Ряд Фурье для f(x) = x + 2sin(3x) на [-π, π]:\n")
print(f"f(x) ≈ {a0 / 2:.3f}")

for n in range(1, N + 1):
    cos_term = f"+ {an[n - 1]:.3f}·cos({n}x)" if abs(an[n - 1]) > 1e-10 else ""
    sin_term = f"+ {bn[n - 1]:.3f}·sin({n}x)" if abs(bn[n - 1]) > 1e-10 else ""

    # Выделение коэффициента для n=3
    if n == 3:
        print(f" {cos_term} + [2.000·sin(3x)] + {bn[n - 1] - 2:.3f}·sin(3x)")
    else:
        print(f" {cos_term} {sin_term}")


# Функция для частичной суммы ряда
def fourier_sum(x, a0, an, bn, N):
    result = a0 / 2
    for n in range(1, N + 1):
        result += an[n - 1] * np.cos(n * x) + bn[n - 1] * np.sin(n * x)
    return result


# Построение
x = np.linspace(-L, L, RESOLUTION)
y_true = f(x)
y_fourier = fourier_sum(x, a0, an, bn, N)

plt.figure(figsize=(14, 7))
plt.plot(x, y_true, label='Исходная функция', linewidth=2, color='lightblue')
plt.plot(x, y_fourier, '--', label=f'Ряд Фурье (N={N})', linewidth=1.5, color='red')

plt.axvline(-np.pi, color='gray', linestyle=':', alpha=0.5)
plt.axvline(np.pi, color='gray', linestyle=':', alpha=0.5)

plt.title('Разложение f(x) = x + 2sin(3x) на [-π, π]', fontsize=14)
plt.xlabel('x (радианы)', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(
    ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    labels=['-π', '-π/2', '0', 'π/2', 'π']
)
plt.show()
