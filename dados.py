import tensorflow as tf
import numpy as np
import os
import scipy
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers, Model

# Definindo os caminhos para os dados
main_path = "/Users/joao/Downloads/chest_xray/chest_xray"
train_path = os.path.join(main_path, "train")
test_path = os.path.join(main_path, "test")
val_path = os.path.join(main_path, "val")

# Função para contar amostras e obter o tamanho da imagem
def get_dataset_info(path):
    total_images = 0
    image_size = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".jpeg") or file.lower().endswith(".jpg") or file.lower().endswith(".png"):
                total_images += 1
                if image_size is None:
                    image = mpimg.imread(os.path.join(root, file))
                    image_size = image.shape
    return total_images, image_size

# Obtenção de informações sobre o conjunto de dados
total_train_images, train_image_size = get_dataset_info(train_path)
total_test_images, test_image_size = get_dataset_info(test_path)
total_val_images, val_image_size = get_dataset_info(val_path)

print("Total de imagens de treino:", total_train_images, "Tamanho das imagens:", train_image_size)
print("Total de imagens de teste:", total_test_images, "Tamanho das imagens:", test_image_size)
print("Total de imagens de validação:", total_val_images, "Tamanho das imagens:", val_image_size)

# Preparando os geradores de dados
train_datagen = ImageDataGenerator(rescale=1.0/255., zoom_range=0.2, shear_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0/255.)
val_datagen = ImageDataGenerator(rescale=1.0/255.)

train_generator = train_datagen.flow_from_directory(train_path, batch_size=16, class_mode='binary', target_size=(150, 150), shuffle=True)
test_generator = test_datagen.flow_from_directory(test_path, batch_size=16, class_mode='binary', target_size=(150, 150))
val_generator = val_datagen.flow_from_directory(val_path, batch_size=16, class_mode='binary', target_size=(150, 150))

# Construindo o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape=(150,150,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # ... Adicione outras camadas conforme necessário
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Saída para classificação binária
])

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Treinando o modelo
train_samples = 5216
batch_size = 16
epochs = 2

history = model.fit(
    train_generator,
    steps_per_epoch = train_samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    verbose=2
)

# Avaliação do modelo
model.evaluate(val_generator)

# InceptionV3 como base para um novo modelo
pretrain_model = InceptionV3(input_shape=(150, 150, 3), weights='imagenet', include_top=False)
for layer in pretrain_model.layers:
    layer.trainable = False

# Construção do modelo transferido
x = layers.Flatten()(pretrain_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)  # Saída para classificação binária

model1 = Model(pretrain_model.input, x) 
model1.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Treinando o modelo transferido
history1 = model1.fit(train_generator, steps_per_epoch=train_samples // batch_size, epochs=epochs, validation_data=test_generator, verbose=2)

# Avaliação do modelo transferido
model1.evaluate(val_generator)

# Predições
predictions = model.predict(val_generator)
x = val_generator.next()
for i in range(8):
    plt.imshow(x[0][i])
    plt.show()
    print("The probability of pneumonia: ", predictions[i])


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
