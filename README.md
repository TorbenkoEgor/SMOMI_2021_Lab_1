## Лабораторная работа №1

При описании нейронных сетей описывались только Conv2D слои. Input, MaxPool2D, Flatten, Dense используются, но не описываются. По умолчанию BATCH_SIZE = 256, lr = 0.001.

---

# train1.py

Модель использует 1 свёрточных слоя. В свёрточном слое 8 фильтров.

build_model:

    inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

Тренировочные данные - Серый график

Валидационные данные - Оранжевый график

Метрика точности
 
![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_1/blob/main/graphs/train1_acc.png)

Метрика функции потерь

![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_1/blob/main/graphs/train1_loss.png)

---

# train2.py

Модель использует 3 свёрточных слоя. В свёрточных слоях 16, 16, 8 фильтров соответсвенно.

build_model

    inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3)(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3)(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

Тренировочные данные - Голубой график

Валидационные данные - Красный график

Метрики точности

![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_1/blob/main/graphs/train2_acc.png)

Метрика функции потерь

![Image alt](https://github.com/TorbenkoEgor/SMOMI_2021_Lab_1/blob/main/graphs/train2_loss.png)

---

# Вывод

Первая реализация нейронной сети не обучилась по причине того, что она состоит только из одного свёрточного слоя. Вторая реализация нейронной сети также не обучилась, а графики потерь начали стремиться к большим значениям более интесивно.
