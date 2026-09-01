
"""FGSM АТАКА"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

print("=" * 60)
print("FGSM АТАКА НА НЕЙРОСЕТЬ")
print("=" * 60)

# Загружаем данные
print("\n1. Загрузка данных MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Нормализация
x_test = x_test.astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1)
y_test_onehot = keras.utils.to_categorical(y_test, 10)

# Создаём и обучаем модель
print("\n2. Создание и обучение модели (30-60 секунд)...")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучаем на уменьшенной выборке для скорости
x_train_small = x_train[:5000]
y_train_small = keras.utils.to_categorical(y_train[:5000], 10)
x_train_small = x_train_small.astype('float32') / 255.0
x_train_small = np.expand_dims(x_train_small, axis=-1)

model.fit(x_train_small, y_train_small,
          batch_size=64,
          epochs=3,
          verbose=1)

# Проверяем точность
test_loss, test_acc = model.evaluate(x_test, y_test_onehot, verbose=0)
print(f"\n Базовая точность модели: {test_acc:.4f} ({test_acc*100:.1f}%)")

# FGSM атака
def fgsm_attack(model, images, labels, epsilon):
    labels_idx = tf.argmax(labels, axis=1)
    images_tensor = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
        tape.watch(images_tensor)
        predictions = model(images_tensor)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels_idx, predictions)
    gradient = tape.gradient(loss, images_tensor)
    signed_grad = tf.sign(gradient)
    adversarial = images_tensor + epsilon * signed_grad
    return tf.clip_by_value(adversarial, 0, 1).numpy()

# Тестируем атаку
print("\n3. Тестирование атаки FGSM...")
num_test = 500
x_sample = x_test[:num_test]
y_sample = y_test_onehot[:num_test]
true_labels = y_test[:num_test]

epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
accuracies = []

for eps in epsilons:
    x_adv = fgsm_attack(model, x_sample, y_sample, eps)
    pred_adv = model.predict(x_adv, verbose=0)
    adv_acc = np.mean(np.argmax(pred_adv, axis=1) == true_labels)
    accuracies.append(adv_acc)
    print(f"   ε={eps:.2f}: точность = {adv_acc:.4f} ({adv_acc*100:.1f}%)")

# Вывод результатов для курсовой
print("\n" + "=" * 60)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ДЛЯ КУРСОВОЙ РАБОТЫ:")
print("=" * 60)
print(f"Базовая точность (без атаки): {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"Точность при ε=0.1: {accuracies[2]:.4f} ({accuracies[2]*100:.1f}%)")
print(f"Снижение точности: {test_acc - accuracies[2]:.4f} ({(test_acc - accuracies[2])*100:.1f}%)")
print("\nВЫВОД: Нейросеть успешно атакована методом FGSM.")
print("=" * 60)
