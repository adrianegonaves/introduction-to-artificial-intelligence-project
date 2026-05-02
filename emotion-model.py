import kagglehub
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


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
    label_mode="categorical",
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

