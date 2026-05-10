import kagglehub
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam

# PATH E VARIÁVEIS GLOBAIS DE CONFIGURAÇÃO
EPOCHS = 30
BATCH = 32
IMAGE_SIZE = (48, 48)
SEED = 50

# CARREGAR O DATASET
PATH = kagglehub.dataset_download('msambare/fer2013')
print("Path to dataset files:", PATH)
# Definir os diretórios de treino e teste usando a variável global "path"
TRAIN_DIR = os.path.join(PATH, "train")
TEST_DIR = os.path.join(PATH, "test")

# VISUALIZAÇÃO DE DADOS 

def plot_dataset_distribution(train_dir, test_dir):

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

    #  Criar visualização comparativa com graficos de barras o plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico do dataset de Treino
    sns.barplot(ax=axes[0], x=list(train_counts.keys()), y=list(train_counts.values()), 
                hue=list(train_counts.keys()), palette="viridis", legend=False)
    axes[0].set_title("Distribuição: Treino")
    axes[0].set_ylabel("Número de Imagens")

    # Gráfico do dataset de Teste
    sns.barplot(ax=axes[1], x=list(test_counts.keys()), y=list(test_counts.values()), 
                hue=list(test_counts.keys()), palette="magma", legend=False)
    axes[1].set_title("Distribuição: Teste")
    axes[1].set_ylabel("Número de Imagens")

    plt.tight_layout()
    plt.show()

    # Impressão dos números de imagens por categoria para ambos os datasets
    print("-" * 30)
    print("RELATORIO DE QUANTIDADES")
    print("-" * 30)
    for emotion in train_counts.keys():
        t_val = train_counts.get(emotion, 0)
        v_val = test_counts.get(emotion, 0)
        print(f"{emotion.capitalize():<10} | Treino: {t_val:>5} | Teste: {v_val:>5}")

    return train_counts

# para visualizar a distribuição das imagens por categoria em ambos os datasets
# Para obervar as imagens, vamos pegar 4 exemplos de cada categoria do dataset de treino. 
def plot_emotion_samples(directory, n_examples=4):
    categories = sorted(os.listdir(directory))
    n_categories = len(categories)
    
    plt.figure(figsize=(12, 2 * n_categories))
    
    for row, emotion in enumerate(categories):
        folder_path = os.path.join(directory, emotion)
        img_names = os.listdir(folder_path)[:n_examples]

        for col, img_name in enumerate(img_names):
            img_path = os.path.join(folder_path, img_name)
            img = plt.imread(img_path)

            plt.subplot(n_categories, n_examples, row * n_examples + col + 1)
            plt.imshow(img, cmap='gray')
            
            if col == 0:
                plt.ylabel(emotion.upper(), fontsize=10, rotation=0, labelpad=40, fontweight='bold')
            
            plt.xticks([])
            plt.yticks([])

    plt.suptitle("Amostras do Dataset (FER-2013)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# Chamada das funções de visualização
counts = plot_dataset_distribution(TRAIN_DIR, TEST_DIR)
plot_emotion_samples(TRAIN_DIR, n_examples=4)

# CARRAGAMENTO E PREPARAÇÃO DOS DADOS PARA O MODELO

# Usando TensorFlow/Keras para garantir que as imagens tenham o tamanho e formato corretos para o modelo, também para separar os dados de treino, validação e teste de forma eficiente.
#Irá devolver um conjunto de dados que produz lotes de 32 imagens dos subdiretórios , juntamente com os rótulos 0 e 10 para cada imagem, dependendo do subdiretório em que se encontram.

def load_data():
    #validação do dataset
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size= BATCH, # usamos o 32 por padrão, recomendando na documentação
        color_mode="grayscale",
        label_mode="categorical",
        labels="inferred",
        validation_split=0.2,
        subset="validation",
        seed=SEED
    )

#################################################################################################################
    # visualização do contéudo com o obetivo de entender melhor a estrutura dos dados e garantir que estão sendo carregados corretamente para o modelo.

    # categorias encontradas
    print("Classes encontradas:", val_ds.class_names)
    # Pegar um lote de 32 imagens
    for imagens, labels in val_ds.take(1):
        print("Formato do lote de imagens:", imagens.shape)  
        # Ver a primeira etiqueta do lote 
        print("Exemplo de etiqueta:", labels[0].numpy())
 #################################################################################################################
    #Treino
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH,
        color_mode="grayscale",
        label_mode="categorical", # "categorical" para classificação multiclasse, onde cada rótulo é representado como um vetor one-hot.
        labels="inferred",
        validation_split=0.2,
        subset="training",
        seed=SEED
    )

    #Teste
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TEST_DIR, 
        image_size=IMAGE_SIZE,
        batch_size=BATCH,
        color_mode="grayscale",
        label_mode="categorical"
    )

# Otimização de performance
    AUTOTUNE = tf.data.AUTOTUNE
    return (train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE), 
            val_ds.cache().prefetch(buffer_size=AUTOTUNE), 
            test_ds.cache().prefetch(buffer_size=AUTOTUNE),
            train_ds.class_names)

train_ds, val_ds, test_ds, class_names = load_data()
# INSPEÇÃO DOS DADOS E NORMALIZAÇÃO

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


# MODELO DE REDE NEURAL CONVOLUCIONAL (CNN) PARA CLASSIFICAÇÃO DE EMOÇÕES

def build_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255), # Normaliza os valores dos pixels para o intervalo [0,1]

        # Layers 01
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(), # Estandariza as ativações da camada anterior, o que pode acelerar o treinamento e melhorar a estabilidade do modelo.
        layers.Activation('relu'), # utiliza o valor máximo entre 0 e a entrada, introduzindo não linearidade no modelo, curvas de separação mais complexas.
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(rate=0.25), # remove da rede 25% dos neurônios para evitar overfitting, de forma aleatória, a cada EPOCHS de treino. Efeta o metodo fit().

        # Layers 02
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(rate=0.25),

        # Layers 03
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
       layers.Dropout(rate=0.25),

        # Layers 04
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(rate=0.25),


        layers.Flatten(),
            
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(rate=0.5),

        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(rate=0.5),

        # Camada de Saída/ output layer
        # Usamos softmax porque label_mode="categorical" (one-hot encoding)
        layers.Dense(units= num_classes, activation='softmax'),
    ])

    # Compilar
    model.compile(
        optimizer= 'Adam', # learning_rate=0.001, # taxa de aprendizado padrão para Adam, parametro.
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Instanciar o modelo
model = build_emotion_model(num_classes=len(class_names))
model.summary()
model.save('modelo_emocoes.h5')


# Para o treino se não houver melhora em 10 épocas
early_stopping = EarlyStopping(
    monitor='val_loss', # val_accuracy ou val_loss, dependendo do que você quer monitorar
    patience=8, 
    restore_best_weights=True
)

# Diminui o Learning Rate se o modelo travar num valor de perda
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=4, 
    min_lr=0.00001
)
model_checkpoint = ModelCheckpoint(
    'best_emotion_model.keras',
     save_best_only=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# VISUALIZAÇÃO DE RESULTADOS 

def plot_training_history(history):
    """
    Gera gráficos de Acurácia e Perda lado a lado.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    # Gráfico de Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Treino', color='#2ecc71', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validação', color='#e74c3c', linewidth=2)
    plt.title('Acurácia do Modelo', fontsize=14)
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Gráfico de Perda (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Treino', color='#2ecc71', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validação', color='#e74c3c', linewidth=2)
    plt.title('Perda (Loss) do Modelo', fontsize=14)
    plt.xlabel('Épocas')
    plt.ylabel('Erro')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, test_ds, class_names):
    y_true = []
    y_pred = []

    for x, y in test_ds:
        predictions = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Relatório de Métricas (Precision, Recall, F1-Score)
    print("\n" + "="*60)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report



def plot_confusion_matrix_and_report(model, test_ds, class_names):
   
    y_true = []
    y_pred = []

    # Coletar todas as predições do dataset de teste
    for x, y in test_ds:
        preds = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão', fontsize=16)
    plt.ylabel('Verdadeiro (True Label)')
    plt.xlabel('Predito (Predicted Label)')
    plt.show()

    # Relatório de Métricas (Precision, Recall, F1-Score)
    print("\n" + "="*60)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))

def display_test_predictions(model, test_ds, class_names, num_samples=6):
   
    for images, labels in test_ds.take(1):
        preds = model.predict(images, verbose=0)
        
        plt.figure(figsize=(15, 5))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(images[i].numpy().squeeze(), cmap='gray')
            
            true_idx = np.argmax(labels[i])
            pred_idx = np.argmax(preds[i])
            
            color = 'green' if true_idx == pred_idx else 'red'
            
            plt.title(f"T: {class_names[true_idx]}\nP: {class_names[pred_idx]}", 
                      color=color, fontsize=10)
            plt.axis('off')
        plt.suptitle("Amostras de Predição (Verde: Correto | Vermelho: Errado)", y=1.05)
        plt.tight_layout()
        plt.show()

# EXECUÇÃO DAS VISUALIZAÇÕES 
plot_training_history(history)
plot_confusion_matrix_and_report(model, test_ds, class_names)
display_test_predictions(model, test_ds, class_names)



