import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from collections import Counter

#Función para calcular la distancia euclidiana entre dos puntos
def calcular_distancia(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Función para inicializar os centroides aleatoriamente
def inicializar_centroides(X, metodo='random'):
    if metodo == 'random':
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        centroids = np.random.uniform(min_vals, max_vals, (4, X.shape[1]))  
    elif metodo == 'dataset':
        centroids = X[np.random.choice(X.shape[0], 4, replace=False)]
    return centroids

# Función para asignar puntos aos centroides máis cercanos
def asignar_cluster(X, centroides):
    if len(centroides) == 0:
        raise ValueError("Error: La lista de centroides está vacía.")
    
    clusters = []
    puntos_asignados = Counter()  
    
    for punto in X:
        distancias = [calcular_distancia(punto, centroide) for centroide in centroides]  

        if len(distancias) == 0 or all(np.isnan(distancias)):  
            raise ValueError("Error: No se pudieron calcular distancias para un punto.")
                
        min_distance = np.nanmin(distancias)
        
        min_indices = [i for i, d in enumerate(distancias) if d == min_distance]

        if not min_indices:  
            raise ValueError("Error: No se encontraron índices con distancia mínima.")
        
        cluster = min(min_indices, key=lambda c: puntos_asignados[c])
        
        clusters.append(cluster)
        puntos_asignados[cluster] += 1  
    
    return np.array(clusters)

# Función para actualizar los centroides
def actualizar_centroides(X, clusters, k):
    nuevos_centroides = np.array([X[clusters == i].mean(axis=0) for i in range(k)])
    return nuevos_centroides    

# Función K-Means
def k_means(X, k=4, metodo='random', max_iter=100):
    centroides = inicializar_centroides(X, metodo=metodo)
    centroides_antiguos = centroides.copy()
    for _ in range(max_iter):
        clusters = asignar_cluster(X, centroides)
        centroides = actualizar_centroides(X, clusters, k)
        
        if np.all(centroides == centroides_antiguos):
            break
        
        centroides_antiguos = centroides.copy()
        
    return centroides, clusters

# Datos de prueba
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Aplicar K-Means
centroides, clusters = k_means(X, k=4, metodo='random', max_iter=10)

# Visualización dos clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=50)
plt.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='x', s=200, label='Centroides')
plt.title("Clusterización K-Means")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()