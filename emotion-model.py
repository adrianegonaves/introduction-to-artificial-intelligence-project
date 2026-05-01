import kagglehub
import os
import matplotlib.pyplot as plt
import seaborn as sns


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
print("RELATÓRIO DE QUANTIDADES")
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

