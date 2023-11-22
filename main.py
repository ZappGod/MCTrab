import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar o conjunto de dados a partir de um arquivo CSV
data_movies = pd.read_csv('imdb_movies_shows.csv')

# Remover linhas com valores ausentes nos campos relevantes
data_movies.dropna(subset=['release_year', 'runtime', 'imdb_score', 'imdb_votes'], inplace=True)

# Redefinir índices após remover linhas
data_movies.reset_index(drop=True, inplace=True)

# Mostrar as primeiras linhas do conjunto de dados para entender a estrutura e os atributos disponíveis
print(data_movies.head())

# Extrair os dados relevantes para prever o resultado das partidas
atributo1 = data_movies['release_year']
atributo2 = data_movies['runtime']
atributo3 = data_movies['imdb_score']

# Normalização dos dados
atributo1 = (atributo1 - atributo1.mean()) / atributo1.std()
atributo2 = (atributo2 - atributo2.mean()) / atributo2.std()
atributo3 = (atributo3 - atributo3.mean()) / atributo3.std()

# Parâmetros da rede
numEpocas = 50
q = len(data_movies)
eta = 0.005
m = 3  # 3 atributos para previsão
N = 8  # Número de neurônios na camada escondida
L = 1

# Inicialização aleatória 
W1 = np.random.random((N, m + 1)) #dimensões da Matriz de entrada
WU = np.random.random((N, N + 1)) #dimensões da Matriz de entrada
WX = np.random.random((N, N + 1)) #dimensões da Matriz de entrada
W2 = np.random.random((L, N + 1)) #dimensões da Matriz de saída

# Vetores de entrada
X = np.vstack((atributo1, atributo2, atributo3))

# Vetor de classificação desejada
d = data_movies['imdb_votes']

# Arrays para armazenar os erros
E = np.zeros(q)
Etm = np.zeros(numEpocas)

# Bias
bias = 1

# Sigmóide
def sigmoid(num):
   return 1/(1+np.exp(-num))

# Treinamento
for i in range(numEpocas):
    for j in range(q):
        # Inserção do bias
        Xb = np.hstack((bias, X[:, j]))

        # Saída da camada escondida
        o1  = sigmoid(W1.dot(Xb)) 
        
        # Incluindo o bias. Saída da camada escondida é a entrada da camada
        # de saída.
        o1b = np.insert(o1, 0, bias)
        o2  = sigmoid(WU.dot(o1b)) 
        o2b = np.insert(o2, 0, bias)          
        o3  = sigmoid(WU.dot(o2b)) 
        o3b = np.insert(o3, 0, bias)  

        # Saída da rede neural
        Y = np.tanh(W2.dot(o1b))

        e = d[j] - Y
        
        # Erro Total.
        E[j] = (e.transpose().dot(e)) / 2

        # Error backpropagation.   
        # Cálculo do gradiente na camada de saída.
        delta2 = np.diag(e).dot((1 - Y*Y))
        vdelta2 = (W2.transpose()).dot(delta2)     
        deltaU = np.diag(1 - o3b*o3b).dot(vdelta2) 
        vdeltaU = (WU.transpose()).dot(deltaU[1:]) 
        deltaX = np.diag(1 - o2b*o2b).dot(vdeltaU)  
        vdeltaX = (WX.transpose()).dot(deltaX[1:])  
        delta1 = np.diag(1 - o1b*o1b).dot(vdeltaX)

        # Atualização dos pesos.
        W1 = W1 + eta*(np.outer(delta1[1:], Xb))
        WU = WU + eta*(np.outer(deltaU[1:], o3b))
        WX = WX + eta*(np.outer(deltaX[1:], o2b))
        W2 = W2 + eta*(np.outer(delta2, o1b))

    Etm[i] = E.mean()

# Plotar o gráfico
plt.xlabel("Épocas")
plt.ylabel("Erro Médio")
plt.plot(Etm, color='b')
plt.show()

# Teste da rede
Error_Test = np.zeros(q)

for i in range(q):
    # Inserção do bias
    Xb = np.hstack((bias, X[:, i]))
    
    # Saída da camada escondida
    o1 = np.tanh(W1.dot(Xb))
    
    # Incluindo o bias. Saída da camada escondida é a entrada da camada
    # de saída.
    o1b = np.insert(o1, 0, bias)
    o2  = sigmoid(WU.dot(o1b)) 
    o2b = np.insert(o2, 0, bias)          
    o3  = sigmoid(WU.dot(o2b)) 
    o3b = np.insert(o3, 0, bias)
    
    # Neural network output
    Y = np.tanh(W2.dot(o1b))
    print(Y)
    
    Error_Test[i] = d[i] - Y

print(Error_Test)
print(np.round(Error_Test))