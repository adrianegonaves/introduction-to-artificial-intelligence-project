import kagglehub
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
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
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        # Normaliza os valores dos pixels para o intervalo [0,1]
        tf.keras.layers.Rescaling(1./255, input_shape=input_shape),

        # Layers 01
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(), # Estandariza as ativações da camada anterior, o que pode acelerar o treinamento e melhorar a estabilidade do modelo.
        tf.keras.layers.Activation('relu'), # utiliza o valor máximo entre 0 e a entrada, introduzindo não linearidade no modelo, curvas de separação mais complexas.
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(rate=0.25), # remove da rede 25% dos neurônios para evitar overfitting, de forma aleatória, a cada EPOCHS de treino. Efeta o metodo fit().

        # Layers 02
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(rate=0.25),

        # Layers 03
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(rate=0.25),

        # Layers 04
        tf.keras.layers.Conv2D(512, (3, 3), padding='same'),
        tf.keras.layers.Conv2D(512, (3, 3), padding='same'),
        tf.keras.layers.Conv2D(512, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(rate=0.25),


        tf.keras.layers.Flatten(),
            
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(rate=0.5),

        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(rate=0.5),

        # Camada de Saída/ output layer
        # Usamos softmax porque label_mode="categorical" (one-hot encoding)
        tf.keras.layers.Dense(units= num_classes, activation='softmax'),
    ])

    return model

# Instanciar o modelo
model = emotion_model()

# Compilar
model.compile(
    optimizer= 'Adam', # learning_rate=0.001, # taxa de aprendizado padrão para Adam, parametro.
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
model.save('modelo_emocoes.h5')

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
plt.xlabel('epochs')
plt.ylabel('Acurácia')
plt.legend(loc='lower right')
plt.show()


plt.plot(epochs, loss, 'r', label='Treino')
plt.plot(epochs, val_loss, 'b', label='Validação')
plt.title('Perda do Modelo')
plt.xlabel('epochs')
plt.ylabel('Perda')
plt.legend(loc='lower right')
plt.show()



class_names = train_ds.class_names
# 1. Pegar um lote de imagens e etiquetas do dataset de teste
for images_batch, labels_batch in test_ds.take(1):
    # Fazer as predições para esse lote
    predictions = model.predict(images_batch)
    
    # Converter as etiquetas de volta para arrays numpy se necessário
    images_batch = images_batch.numpy()
    labels_batch = labels_batch.numpy()
    
    # Como as imagens estão em escala de cinza (48, 48, 1), 
    # precisamos remover a última dimensão para o plt.imshow exibir corretamente
    images_display = images_batch.squeeze()
    
    break # Só precisamos de um lote para a visualização
def display_predictions(images, y_true, y_pred, class_names, num_samples=5):
    plt.figure(figsize=(10, 6))

    num_samples = min(num_samples, len(images))

    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap='gray')

        true_label_idx = np.argmax(y_true[i])
        true_label = class_names[true_label_idx]

        pred_label_idx = np.argmax(y_pred[i])
        predicted_label = class_names[pred_label_idx]

        color = 'green' if true_label_idx == pred_label_idx else 'red'

        plt.xlabel(f'True: {true_label}\nPred: {predicted_label}')
    plt.tight_layout()
    plt.show()
# display_predictions

display_predictions(
   images_display, 
    labels_batch, 
    predictions, 
    class_names, 
    num_samples=6
)


import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# 1. Pegar as imagens e labels reais do test_ds
y_true = []
y_pred = []

for x, y in test_ds:
    predictions = model.predict(x)
    y_true.extend(np.argmax(y, axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

# 2. Gerar o relatório
print(classification_report(y_true, y_pred, target_names=categorias))


#----
# Pega apenas 1 lote (batch) do dataset
for imagens, labels in train_ds.take(1):
    # Converte para numpy para facilitar a visualização
    img_exemplo = imagens[0].numpy()
    
    print("--- VALORES ANTES DA NORMALIZAÇÃO ---")
    print(f"Tipo do dado: {img_exemplo.dtype}")
    print(f"Valor Máximo: {img_exemplo.max()}")
    print(f"Valor Mínimo: {img_exemplo.min()}")
    # Imprime uma pequena parte da matriz de pixels (ex: 5x5)
    print("Amostra de pixels:")
    print(img_exemplo[0:5, 0:5, 0])

    # Criar uma camada de escala isolada
scaler = tf.keras.layers.Rescaling(1./255)

# Passar a mesma imagem pela camada
img_normalizada = scaler(imagens[0])

print("\n--- VALORES DEPOIS DA NORMALIZAÇÃO ---")
print(f"Tipo do dado: {img_normalizada.dtype}")
print(f"Valor Máximo: {img_normalizada.numpy().max()}")
print(f"Valor Mínimo: {img_normalizada.numpy().min()}")
print("Amostra de pixels (agora entre 0 e 1):")
print(img_normalizada.numpy()[0:5, 0:5, 0])