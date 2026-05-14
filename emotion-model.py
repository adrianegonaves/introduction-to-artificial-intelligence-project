import kagglehub
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

# PATH E VARIÁVEIS GLOBAIS DE CONFIGURAÇÃO
EPOCHS = 40
#EPOCHS = 20
#EPOCHS = 10
#BATCH = 32
BATCH = 64 #deixei 64 para acelerar o processo de treino, mas dependendo do hardware disponível, pode ser necessário ajustar para 32 ou outro valor.
IMAGE_SIZE = (48, 48)
SEED = 50

# CARREGAR O DATASET
PATH = kagglehub.dataset_download('msambare/fer2013')
print("Path to dataset files:", PATH)
# Definir os diretórios de treino e teste usando a variável global "path"
TRAIN_DIR = os.path.join(PATH, "train")
TEST_DIR = os.path.join(PATH, "test")

######################################################
# VISUALIZAÇÃO DE DADOS 
######################################################

#função auxilar para contar o número de imagens em cada categoria de emoção, tanto para o dataset de treino quanto para o de teste. Ele percorre os subdiretórios dentro do caminho base fornecido e conta quantos arquivos (imagens) existem em cada um, retornando um dicionário com as contagens.
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

# função para plotar a distribuição das imagens por categoria para os datasets de treino e teste, usando gráficos de barras para facilitar a comparação visual entre os dois conjuntos de dados. Ele também imprime um relatório detalhado com o número de imagens em cada categoria para ambos os datasets.
def plot_dataset_distribution(train_dir, test_dir):
    # Obter dados de ambos os diretórios
    train_counts = get_counts(train_dir)
    test_counts = get_counts(test_dir)

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

    #Impressão dos números de imagens por categoria para ambos os datasets
    print("-" * 30)
    print("RELATORIO DE QUANTIDADES")
    print("-" * 30)
    for emotion in train_counts.keys():
        t_val = train_counts.get(emotion, 0)
        v_val = test_counts.get(emotion, 0)
        print(f"{emotion.capitalize():<10} | Treino: {t_val:>5} | Teste: {v_val:>5}")

    return train_counts

#Função para imprimir uma grelha de imagens, onde cada linha representa uma categoria de emoção e cada coluna mostra um exemplo diferente dessa categoria. Ele percorre os subdiretórios do diretório fornecido, seleciona um número específico (4) de imagens de cada categoria e as exibe usando Matplotlib, com rótulos indicando a emoção correspondente.
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

######################################################
# CARRAGAMENTO E PREPARAÇÃO DOS DADOS PARA O MODELO
######################################################
# Usando TensorFlow/Keras para garantir que as imagens tenham o tamanho e formato corretos para o modelo, também para separar os dados de treino, validação e teste de forma eficiente.
#Irá devolver um conjunto de dados que produz lotes de 32 ( ou o tamanho definido ) imagens dos subdiretórios, juntamente com os rótulos 0 e 1 para cada imagem, dependendo do subdiretório em que se encontram. O parâmetro "validation_split=0.2" indica que 20% dos dados serão reservados para validação, e "subset='training'" ou "subset='validation'" especifica se o conjunto de dados retornado deve conter os dados de treinamento ou de validação, respectivamente.
def load_data():
    #Treino
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TRAIN_DIR, # diretório base onde estão as subpastas de cada categoria de emoção.
        image_size=IMAGE_SIZE, #redimensiona as imagens para o tamanho especificado (48x48 pixels) para garantir que todas as imagens tenham a mesma dimensão.
        batch_size=BATCH, # define o número de imagens que serão processadas em cada lote durante o treinamento do modelo.
        color_mode="grayscale", # garante que as imagens sejam carregadas em escala de cinza
        label_mode="categorical", # "categorical" para classificação multiclasse, onde cada rótulo é representado como um vetor one-hot.
        labels="inferred", # os rótulos são inferidos a partir dos nomes das subpastas, ou seja, cada subpasta é considerada uma classe diferente.
        validation_split=0.2, # garate que 20% dos dados sejam reservados para validação
        subset="training", # especifica que este conjunto de dados deve conter os dados de treinamento ou validação(80%)
        seed=SEED # para garantir que a divisão entre treino e validação seja consistente.
    )
    
    #################################################################################################################
    # visualização do contéudo com o obetivo de entender melhor a estrutura dos dados e garantir que estão sendo carregados corretamente para o modelo.Essa parte do codigo é apenas para estudo e não tem impacto no modelo.
    print("-" * 30)
    print("VALIDAÇÃO DO CONTEÚDO DO DATASET")
    print("-" * 30)
    print("Classes encontradas:", train_ds.class_names)
    # Pegar um lote de 32 imagens
    for imagens, labels in train_ds.take(1):
        print("Formato do lote de imagens:", imagens.shape)  
        # Ver a primeira etiqueta do lote 
        print("Exemplo de etiqueta:", labels[0].numpy())
    #################################################################################################################
 
   #validação do dataset
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size= BATCH, 
        color_mode="grayscale",
        label_mode="categorical",
        labels="inferred",
        validation_split=0.2,
        subset="validation",
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

#################################################################################################################
#Mais um etapa do codigo criada apenas com o objetivo de estudar e entender o funcionamento. Queriamos entender melhor o conceito de normalização e como os valores dos pixels são transformados antes de serem alimentados no modelo, pois a normalizalão é uma etapa importante. Portando essa parte não tenho impacto no modelo.

# INSPEÇÃO DOS DADOS E NORMALIZAÇÃO
print("-" * 30)
print("INSPEÇÃO DOS DADOS E NORMALIZAÇÃO")
print("-" * 30)

for imagens, labels in train_ds.take(1):
# Converte para numpy para facilitar a visualização
    example_image = imagens[0].numpy()

print("--- VALORES ANTES DA NORMALIZAÇÃO ---")
print(f"Tipo do dado: {example_image.dtype}")
print(f"Valor Máximo: {example_image.max()}")
print(f"Valor Mínimo: {example_image.min()}")
# Imprime uma pequena parte da matriz de pixels
print("Amostra de pixels:")
print(example_image[0:5, 0:5, 0])

# Criar uma camada de escala isolada e normaliza. 
scaler = tf.keras.layers.Rescaling(1./255)
# Passar a mesma imagem pela camada
image_normalizada = scaler(imagens[0])

print("\n--- VALORES DEPOIS DA NORMALIZAÇÃO ---")
print(f"Tipo do dado: {image_normalizada.dtype}")
print(f"Valor Máximo: {image_normalizada.numpy().max()}")
print(f"Valor Mínimo: {image_normalizada.numpy().min()}")
print("Amostra de pixels (agora entre 0 e 1):")
print(image_normalizada.numpy()[0:5, 0:5, 0])

#################################################################################################################

# Função para criar o modelo de rede neural convolucional (CNN) para classificação de emoções, recebe como argumento o numero de classes e o input_shape (que é o formato das imagens de entrada, neste caso, 48x48 pixels em escala de cinza) e retorna um modelo compilado.
def build_emotion_model(input_shape=(48, 48, 1), number_classes= 7):
    model = models.Sequential([
        # Camada de entrada e normalização
        layers.Input(shape=input_shape),# Define a forma dos dados de entrada para o modelo, neste caso, imagens de 48x48 pixels em escala de cinza (1 canal).
        layers.Rescaling(1./255), # Normaliza os valores dos pixels para o intervalo [0,1]

        # Bloco 01
        layers.Conv2D(32, (3, 3), padding='same'), # camada convolucional que aplica 32 filtros de tamanho 3x3 às imagens de entrada, com preenchimento 'same' par manter o tamanho da imagem igual à entrada, evitando que as bordas sejam ignoradas.
        layers.BatchNormalization(), # Estandariza as ativações da camada anterior, o que pode acelerar o treinamento e melhorar a estabilidade do modelo.
        layers.Activation('relu'), # introduz não linearidade no modelo, curvas de separação mais complexas.
        layers.MaxPooling2D(pool_size=(2, 2)), # adiciona uma camada de pooling que reduzirá a imagem pela metade de sua altura e largura originais.
        layers.Dropout(rate=0.2), # remove da rede 20% dos neurônios para evitar overfitting, de forma aleatória, a cada EPOCHS de treino. Efeta o metodo fit().

        # Bloco 02
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(rate=0.2),

        # Bloco 03
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(rate=0.2),

     
        layers.Flatten(), # transforma as saídas 2D das camadas convolucionais anteriores em um vetor 1D, que pode ser alimentado em camadas densas totalmente conectadas.
            
        layers.Dense(256), # camada densa totalmente conectada com 256 neurônios, que processa as características extraídas pelas camadas convolucionais anteriores.
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(rate=0.2),
        
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(rate=0.2),

        # Camada de Saída/ output layer
        # Usamos softmax porque label_mode="categorical" (one-hot encoding), que  ter o formato binario para cada classe, e softmax é a função de ativação apropriada para problemas de classificação multiclasse, pois converte as saídas do modelo em probabilidades que somam 1, facilitando a interpretação dos resultados.
        layers.Dense(number_classes, activation='softmax'),
    ])

    # Compilar
    model.compile(
        optimizer=Adam(learning_rate=0.001), # learning_rate=0.001
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Instanciar o modelo com o número de classes igual ao número de categorias de emoções presentes no dataset, que é determinado pela variável class_names.
model = build_emotion_model(number_classes=len(class_names))
#comando summary() mostra algumas informações sobre as camadas do modelo. Podemos ver as dimensões de cada camada e os parâmetros aprendidos em cada etapa.
model.summary()

##Callbacks para o treinamento do modelo:
# À medida que o modelo aprende, a perda nos dados de treinamento tende a diminuir, mas a perda nos dados de validação pode começar a aumentar se o modelo começar o overfitting dos dados de treinamento. O EarlyStopping monitora a perda de validação e, se ela não melhorar por um número especificado de epochs (neste caso, 15).
early_stopping = EarlyStopping(
    monitor='val_loss', # monitora a perda de validação para determinar quando parar o treinamento.
    patience=15,
    restore_best_weights=True # garante que, quando o treinamento for interrompido, os pesos do modelo sejam restaurados para os valores correspondentes à melhor perda de validação observada durante o treinamento.
)
# Durante o treinamento, se a perda de validação não melhorar por um número especificado de epochs (neste caso 5), a taxa de aprendizado será reduzida. 
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, #Fator pelo qual a taxa de aprendizado será reduzida
    patience=5, #Se a perda de validação não melhorar por 5 epochs, a taxa de aprendizado será reduzida.
    min_lr=0.00001 # define um limite inferior para a taxa de aprendizado, garantindo que ela não seja reduzida a um valor tão baixo que o modelo pare de aprender.
)

#Função de retorno de chamada para salvar o modelo Keras 
model_checkpoint = ModelCheckpoint(
    'emotion_model.keras',
     save_best_only=True
)
# Para treinar o modelo, usando os conjuntos de dados de treino e validação, e aplicando os callbacks definidos para monitorar o desempenho do modelo durante o treinamento. O resultado do treinamento é armazenado na variável "history", que contém informações sobre a perda e a acurácia do modelo em cada epoch, tanto para os dados de treinamento quanto para os dados de validação.
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

#########################################################
# VISUALIZAÇÃO DE RESULTADOS 
#########################################################
# Função para plotar os gráficos de Acurácia e Perda (Loss) do modelo durante o treinamento
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
    plt.xlabel('Epochs')
    plt.ylabel('Acurácia')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Gráfico de Perda (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Treino', color='#2ecc71', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validação', color='#e74c3c', linewidth=2)
    plt.title('Perda (Loss) do Modelo', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Erro')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

#Função para gerar a matriz de confusão e o relatório de classificação, que inclui métricas como precisão, recall e F1-score para cada categoria de emoção. 
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
    print("RELATORIO DE CLASSIFICACAO")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
# Função para exibir algumas amostras de imagens do dataset de teste, juntamente com suas etiquetas verdadeiras e as predições/previsão do modelo. As predições corretas são destacadas em verde, enquanto as incorretas são destacadas em vermelho, facilitando a visualização do desempenho do modelo em casos específicos.
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



