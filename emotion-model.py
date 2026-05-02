import kagglehub
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Rescaling
from keras.optimizers import Adam

# Download do dataset

PATH = kagglehub.dataset_download('msambare/fer2013')
print("Path to dataset files:", PATH)

# Definir os diretórios de treino e teste usando a variável global "path"
TRAIN_DIR = os.path.join(PATH, "train")
TEST_DIR = os.path.join(PATH, "test")

# PRÉ-PROCESSAMENTO

# Função para contar imagens em cada subpasta
def get_counts(base_path):
    counts = {}
    if not os.path.exists(base_path):
        return counts
    for emotion in os.listdir(base_path):
        emotion_path = os.path.join(base_path, emotion)
        if os.path.isdir(emotion_path):
            counts[emotion] = len(os.listdir(emotion_path))
    # Retorna ordenado por valor para o gráfico ficar simétrico
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

# Obter dados de ambos os diretórios
train_counts = get_counts(TRAIN_DIR)
test_counts = get_counts(TEST_DIR)

#  Criar visualização comparativa com o plt
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico de Treino
sns.barplot(ax=axes[0], x=list(train_counts.keys()), y=list(train_counts.values()), 
            hue=list(train_counts.keys()), palette="viridis", legend=False)
axes[0].set_title("Distribuição: Treino")
axes[0].set_ylabel("Número de Imagens")

# Gráfico de Teste
sns.barplot(ax=axes[1], x=list(test_counts.keys()), y=list(test_counts.values()), 
            hue=list(test_counts.keys()), palette="magma", legend=False)
axes[1].set_title("Distribuição: Teste")
axes[1].set_ylabel("Número de Imagens")

plt.tight_layout()
plt.show()

# Impressão dos valores no console
print("-" * 30)
print("RELATORIO DE QUANTIDADES")
print("-" * 30)
for emotion in train_counts.keys():
    t_val = train_counts.get(emotion, 0)
    v_val = test_counts.get(emotion, 0)
    print(f"{emotion.capitalize():<10} | Treino: {t_val:>5} | Teste: {v_val:>5}")


categorias = os.listdir(TRAIN_DIR)
# Para obervar as imagens, vamos pegar 4 exemplos de cada categoria do dataset de treino. 
n_exemplos = 4 

plt.figure(figsize=(12, 14))
contador = 1

for emotion in categorias:
    folder_path = os.path.join(TRAIN_DIR, emotion)
    img_names = os.listdir(folder_path)[:n_exemplos]

    for i, img_name in enumerate(img_names):
        img_path = os.path.join(folder_path, img_name)
        img = plt.imread(img_path)

        plt.subplot(len(categorias), n_exemplos, contador)

        plt.imshow(img, cmap='gray')

        if i == 0:
            plt.ylabel(emotion, fontsize=12, rotation=0, labelpad=40)

        plt.xticks([])
        plt.yticks([])

        contador += 1

plt.tight_layout()
plt.show()

# Usando TensorFlow/Keras para garantir que as imagens tenham o tamanho e formato corretos para o modelo, também para separar os dados de treino, validação e teste de forma eficiente.
#Irá devolver um conjunto de dados que produz lotes de de 32 imagens dos subdiretórios , juntamente com os rótulos 0 e 10 para cada imagem, dependendo do subdiretório em que se encontram.

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=TRAIN_DIR,
    image_size=(48, 48),
    batch_size=32, # usamos o 32 por padrão, recomendando na documentação
    color_mode="grayscale",
    label_mode="categorical",
    labels="inferred",
    validation_split=0.2,
    subset="validation",
    seed=50
)

# categorias encontradas
print("Classes encontradas:", val_ds.class_names)

# Pegar um lote de 32 imagens
for imagens, labels in val_ds.take(1):
    print("Formato do lote de imagens:", imagens.shape)  
    
    # Ver a primeira etiqueta do lote 
    print("Exemplo de etiqueta:", labels[0].numpy())

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=TRAIN_DIR,
    image_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    label_mode="categorical", # "categorical" para classificação multiclasse, onde cada rótulo é representado como um vetor one-hot.
    labels="inferred",
    validation_split=0.2,
    subset="training",
    seed=50
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=TEST_DIR, 
    image_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    label_mode="categorical"
)

# MODELO DE REDE NEURAL CONVOLUCIONAL (CNN) PARA CLASSIFICAÇÃO DE EMOÇÕES
def emotion_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential()

    model.add(Rescaling(1./255, input_shape=input_shape)) # Normaliza os valores dos pixels para o intervalo [0,1]

    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten()),
        
    model.add(Dense(256)),
    model.add(BatchNormalization()),
    model.add(Activation('relu')),
    model.add(Dropout(0.5)),

    model.add(Dense(512)),
    model.add(BatchNormalization()),
    model.add(Activation('relu')),
    model.add(Dropout(0.5)),

    # Camada de Saída
    # Usamos softmax porque label_mode="categorical" (one-hot encoding)
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# Instanciar o modelo
model = emotion_model()

# Compilar
model.compile(
    optimizer= Adam(learning_rate=0.001), # learning_rate=0.001, # taxa de aprendizado padrão para Adam
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

EPOCHS = 30
BATCH = 32

# Para o treino se não houver melhora em 10 épocas
early_stopping = EarlyStopping(
    monitor='val_loss', # val_accuracy ou val_loss, dependendo do que você quer monitorar
    patience=10, 
    restore_best_weights=True
)

# Diminui o Learning Rate se o modelo travar num valor de perda
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=5, 
    min_lr=0.00001
)


 # Defina um número alto, o EarlyStopping cuidará de parar antes se necessário

history = model.fit(
    train_ds,
    batch_size=BATCH,
    validation_data=val_ds,
    epochs=EPOCHS,
    shuffle=True,
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy) )
# Gráfico de Acurácia
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia do Modelo')
plt.legend()
plt.show()

plt.plot(epochs, accuracy, 'r', label='Treino')
plt.plot(epochs, val_accuracy, 'b', label='Validação')
plt.title('Acurácia do Modelo')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend(loc='lower right')
plt.show()


plt.plot(epochs, loss, 'r', label='Treino')
plt.plot(epochs, val_loss, 'b', label='Validação')
plt.title('Perda do Modelo')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend(loc='lower right')
plt.show()

model.save('modelo_emocoes.h5')