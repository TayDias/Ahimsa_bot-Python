import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

# Alterar para 1 após a primeira execução
test = 0

with open("intents.json") as file:
    data = json.load(file)

# Se conseguir ler as variaveis: Carrega-las
if test == 1:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

# Se não conseguir ler as variaveis: Preparar o modelo para treino
else:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern) # Separar todas as palavras
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    # Remover palavras duplicadas e somar o total do vocabulário
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels) # Ordenação

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    # Todas as palavras de um padrão enumeradas no padrão: [0,0,1,0]
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:] # Copia
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # tflearn requer arrays ao invés de listas
    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


tensorflow.reset_default_graph()

#Rede Neural
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net) # Tipo de Neuronio

# Carregar o modelo
if test == 1:
    model.load("model.tflearn")

else: # Treinar o modelo
    print("Inicio do teinamento do modelo")

    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

    print("Fim do teinamento do modelo. Mude a variavel test para 1 para testa-lo!")
    

# PREDIÇÕES
# Transformar as sentenças dos usuários no padrão enumerado, ex: [0,0,1,0]
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))] # Muda o valor se a palavra existir

    user_words = nltk.word_tokenize(s)
    user_words = [stemmer.stem(word.lower()) for word in user_words]

    for se in user_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return numpy.array(bag)

def chat():
    print("Incio de conversa com o bot (Para parar, digite: sair)!")
    while True:
        inp = input("Você: ")
        if inp.lower() == "sair":
            break

        # Transformar em bag, alimentar o modelo
        results = model.predict([bag_of_words(inp, words)])

        # Pegar a resposta mais provavel na lista
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print("Bot: "+random.choice(responses))    

if test == 1:
    chat()