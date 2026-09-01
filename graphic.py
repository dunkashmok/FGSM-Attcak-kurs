
"""График для FGSM атаки - запускать после основного кода"""

import matplotlib.pyplot as plt

# Данные из прошлого запуска (если переменные не сохранились - переопределим)
# Но если ты запускал основной код в этой же сессии, переменные уже есть

print("=" * 60)
print("ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ КУРСОВОЙ РАБОТЫ")
print("=" * 60)

# Проверяем, есть ли переменные, если нет - запрашиваем ввод
try:
    epsilons
    accuracies
    test_acc
except NameError:
    print("\n Переменные не найдены. Введи данные вручную:")
    print("(это случилось, если ты запускал график отдельно от обучения)")

    # Ручной ввод (как запасной вариант)
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    print(f"\nСилы атаки (ε): {epsilons}")

    print("\nВведи точность для каждого ε (через пробел):")
    print("Пример: 0.95 0.85 0.65 0.45 0.30 0.20 0.15")
    acc_input = input("Точности: ").split()
    accuracies = [float(x) for x in acc_input]

    print("\nВведи базовую точность модели:")
    test_acc = float(input("Базовая точность: "))

# ============================================================
# ГРАФИК 1: Зависимость точности от силы атаки
# ============================================================

plt.figure(figsize=(10, 6))

# Основная линия
plt.plot(epsilons, accuracies, 'ro-', linewidth=2.5, markersize=10, label='Точность модели после атаки')

# Горизонтальная линия базовой точности
plt.axhline(y=test_acc, color='g', linestyle='--', linewidth=2, label=f'Базовая точность ({test_acc*100:.1f}%)')

# Заливка области снижения
plt.fill_between(epsilons, accuracies, test_acc, alpha=0.3, color='red', label='Снижение точности')

# Настройки
plt.xlabel('Сила атаки ε (epsilon)', fontsize=14, fontweight='bold')
plt.ylabel('Точность классификации', fontsize=14, fontweight='bold')
plt.title('FGSM АТАКА: Зависимость точности от силы возмущения', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(0, 1.05)
plt.xlim(-0.01, 0.31)

# Подписи точек
for eps, acc in zip(epsilons, accuracies):
    plt.annotate(f'{acc:.2f}',
                 (eps, acc),
                 textcoords="offset points",
                 xytext=(0, 15),
                 ha='center',
                 fontsize=10,
                 fontweight='bold')

# Легенда
plt.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('fgsm_accuracy_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# ГРАФИК 2: Пример атаки (3 картинки: оригинал, шум, атакованный)
# ============================================================

print("\n2. Построение визуализации атаки на одном примере...")

# Выбираем один тестовый пример
idx = 5 #                   меняем индекс числа в датасете
epsilon_demo = 0.1

# Берём одну картинку
single_image = x_sample[idx:idx+1]
single_label = y_sample[idx:idx+1]
true_label = true_labels[idx]

# Генерируем атакованный пример
x_adv_single = fgsm_attack(model, single_image, single_label, epsilon_demo)

# Предсказания
orig_pred = np.argmax(model.predict(single_image, verbose=0))
adv_pred = np.argmax(model.predict(x_adv_single, verbose=0))

# Создаём три картинки
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Оригинал
axes[0].imshow(single_image[0].reshape(28, 28), cmap='gray')
axes[0].set_title(f'Оригинал\nИстинная: {true_label}\nПредсказание: {orig_pred}', fontsize=12)
axes[0].axis('off')

# Шум (увеличенный для наглядности)
noise = x_adv_single[0] - single_image[0]
axes[1].imshow(noise.reshape(28, 28), cmap='seismic', vmin=-0.3, vmax=0.3)
axes[1].set_title(f'Шум (увеличен)\nε = {epsilon_demo}', fontsize=12)
axes[1].axis('off')

# Атакованный
axes[2].imshow(x_adv_single[0].reshape(28, 28), cmap='gray')
axes[2].set_title(f'Атакованный\nИстинная: {true_label}\nПредсказание: {adv_pred}', fontsize=12)
axes[2].axis('off')

plt.suptitle('FGSM АТАКА: Пример соревновательного примера', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fgsm_example.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# ГРАФИК 3: Таблица результатов (текстовая)
# ============================================================

print("\n" + "=" * 70)
print(" ТАБЛИЦА РЕЗУЛЬТАТОВ ДЛЯ КУРСОВОЙ РАБОТЫ")
print("=" * 70)

print("\n┌────────────┬──────────────┬─────────────────────┐")
print("│    ε       │  Точность    │  Снижение точности  │")
print("├────────────┼──────────────┼─────────────────────┤")

for i, eps in enumerate(epsilons):
    drop = (test_acc - accuracies[i]) * 100
    print(f"│ {eps:5.2f}      │   {accuracies[i]*100:5.1f}%     │      {drop:5.1f}%         │")

print("└────────────┴──────────────┴─────────────────────┘")


print("\n" + "=" * 60)
print(" ВСЕ ГРАФИКИ ПОСТРОЕНЫ")
print("=" * 60)
