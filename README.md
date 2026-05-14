# Projeto: Introdução à Inteligência Artificial

## 1. Contextualização
Este projeto foi desenvolvido para a unidade curricular de **Introdução à Inteligência Artificial** do curso de Engenharia Informática do **Instituto Politécnico de Santarém**, sob a orientação do **Professor Artur Marques**.

## 2. Equipa
* **Adriane Gonçalves** - 240000004
* **Bruno Hortelão** - 240001083

## 3. Dataset e Metodologia
Utilizámos o dataset **FER-2013** (*Facial Expression Recognition*), que está estruturado em diretórios de treino (`train`) e teste (`test`).

### 3.1 Pré-processamento e Análise Exploratória
A fase inicial do projeto focou-se no pré-processamento e na compreensão dos dados. Realizámos uma análise comparativa entre os conjuntos de treino e teste para:
* Visualizar a tipologia das imagens;
* Verificar a distribuição quantitativa das imagens em cada categoria de emoção.


## 4. Como rodar o projeto:
### 4.1. Configurar o ambiente virtual:

````
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
````
#### 4.2. Instalar dependências
````
pip install -r requirements.txt
````
#### 4.3. Treinar o modelo
````
python emotion-model.py
````
#### 4.4. Executar a aplicação Streamlit
````
streamlit run app.py
````


### 5. Fontes de Consulta:
* **Artigos Técnicos:**
    * [Loading Image Data in Keras: image_dataset_from_directory vs flow_from_dataframe](https://medium.com/@drubodevpramanik/loading-image-data-in-keras-image-dataset-from-directory-vs-flow-from-dataframe-a11890d82e55)
    * [Maximizando a potência do seu modelo de Deep Learning](https://www.linkedin.com/pulse/maximizando-pot%C3%AAncia-do-seu-modelo-de-deep-learning-com-fl%C3%A1via-gaia-/)
    * [Image data loading](https://keras.io/2/api/data_loading/image/)
    * [tf.keras.preprocessing.image_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory)
    * [Normalização em lote: Teoria e implementação do TensorFlow](https://www.datacamp.com/pt/tutorial/batch-normalization-tensorflow)
    * [Uma introdução às redes neurais convolucionais (CNNs)](https://www.datacamp.com/pt/tutorial/introduction-to-convolutional-neural-networks-cnns)
    * [Tutorial de redes neurais convolucionais (CNN) com TensorFlow](https://www.datacamp.com/pt/tutorial/cnn-tensorflow-python)
    * [Aprenda a Criar e Treinar Uma Rede Neural Convolucional (CNN)](https://www.insightlab.ufc.br/aprenda-a-criar-e-treinar-uma-rede-neural-convolucional-cnn/)
    * [Entendendo Redes Convolucionais (CNNs)](https://medium.com/neuronio-br/entendendo-redes-convolucionais-cnns-d10359f21184)
    * [O que são redes neurais convolucionais?](https://www.ibm.com/br-pt/think/topics/convolutional-neural-networks?regionCode=br&languageCode=pt&contactmodule=true&cm-history=br-pt)
    * [Convolutional Neural Network (CNN) in Deep Learning](https://www.geeksforgeeks.org/deep-learning/convolutional-neural-network-cnn-in-machine-learning/)
    * [Como avaliar seu modelo de classificação](https://medium.com/data-hackers/como-avaliar-seu-modelo-de-classificação-34e6f6011108)
    * [Keras](https://keras.io/*)

* **Vídeo Aulas:**
    * [Tutorial de Redes Neuronais (YouTube)](https://www.youtube.com/watch?v=bp8OpasGtV4)

     
    