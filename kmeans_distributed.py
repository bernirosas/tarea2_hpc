import numpy as np
from numba import njit, prange
from mpi4py import MPI



def generate_distributed_data(n_total, d, k, seed):
    """
    Genera datos distribuidos entre procesos MPI.
    Cumple con especificación técnica: generación distribuida sin Scatter inicial.
    
    Args:
        n_total: Número total de muestras
        d: Dimensionalidad (features)
        k: Número de clusters
        seed: Semilla base para reproducibilidad
    
    Returns:
        data: Array local de datos (n_local, d)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Calcular cuántos datos corresponden a este proceso
    n_local = n_total // size
    if rank < n_total % size:
        n_local += 1
    
    # Usar semilla única para cada proceso (seed + rank)
    np.random.seed(seed + rank)
    
    # Generar centroides sintéticos (solo para generación de datos)
    cluster_centers = np.random.randn(k, d) * 10
    
    # Generar datos alrededor de los centroides
    data = np.zeros((n_local, d), dtype=np.float64)
    cluster_assignments = np.random.randint(0, k, size=n_local)
    
    for i in range(n_local):
        cluster_id = cluster_assignments[i]
        # Añadir ruido gaussiano alrededor del centroide
        data[i] = cluster_centers[cluster_id] + np.random.randn(d) * 2.0
    
    return data


# =============================================================================
# 2. Cálculo de Distancias con Numba
# =============================================================================
# Usa @njit(parallel=True) y prange para calcular distancias euclidianas de 
# forma paralela con threads.

@njit(parallel=True)
def compute_distances(data, centroids):
    """
    Calcula distancias euclidianas entre datos y centroides.
    
    Args:
        data: Array (n, d) de puntos
        centroids: Array (k, d) de centroides
    
    Returns:
        distances: Array (n, k) de distancias
    """
    n = data.shape[0]
    k = centroids.shape[0]
    d = data.shape[1]
    
    distances = np.zeros((n, k), dtype=np.float64)
    
    # Paralelización sobre los puntos
    for i in prange(n):
        for j in range(k):
            dist_sq = 0.0
            for dim in range(d):
                diff = data[i, dim] - centroids[j, dim]
                dist_sq += diff * diff
            distances[i, j] = np.sqrt(dist_sq)
    
    return distances


# =============================================================================
# 3. Asignación de Etiquetas con Numba
# =============================================================================
# Asigna cada punto al cluster más cercano basándose en las distancias calculadas.

@njit(parallel=True)
def assign_labels(distances):
    """
    Asigna cada punto al cluster más cercano.
    
    Args:
        distances: Array (n, k) de distancias obtenido de a.ii.2.
    
    Returns:
        labels: Array (n,) con índices del cluster más cercano
    """
    n = distances.shape[0]
    labels = np.zeros(n, dtype=np.int32)
    
    # Paralelización sobre los puntos
    for i in prange(n):
        min_dist = distances[i, 0]
        min_idx = 0
        
        for j in range(1, distances.shape[1]):
            if distances[i, j] < min_dist:
                min_dist = distances[i, j]
                min_idx = j
        
        labels[i] = min_idx
    
    return labels


# =============================================================================
# 4. Cómputo de Sumas Locales con Numba
# =============================================================================
# Acumula las sumas de puntos y conteos por cluster de forma local en cada 
# proceso. Estas sumas se reducirán globalmente usando MPI.

@njit(parallel=True)
def compute_local_sums(data, labels, k):
    """
    Calcula sumas locales de puntos por cluster y conteos.
    
    Args:
        data: Array (n, d) de puntos
        labels: Array (n,) de asignaciones de cluster
        k: Número de clusters
    
    Returns:
        sums: Array (k, d) con suma de puntos por cluster
        counts: Array (k,) con número de puntos por cluster
    """
    n = data.shape[0]
    d = data.shape[1]
    
    sums = np.zeros((k, d), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    
    # Acumular sumas por cluster
    # Nota: No se puede paralelizar directamente con prange aquí
    # debido a condiciones de carrera en las escrituras a sums y counts
    for i in range(n):
        cluster_id = labels[i]
        counts[cluster_id] += 1
        for dim in range(d):
            sums[cluster_id, dim] += data[i, dim]
    
    return sums, counts


# =============================================================================
# 5. Implementación de K-means Distribuido
# =============================================================================
# Implementación del algoritmo K-means usando las funciones anteriores.
# Para ejecutar con MPI: mpirun -n <num_procesos> python kmeans_distributed.py

def kmeans_distributed():
    """
    Implementación completa del algoritmo K-means distribuido.
    Ejecutar con: mpirun -n <num_procesos> python kmeans_distributed.py
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Parámetros del algoritmo (según especificaciones técnicas)
    n_total = 4_000_000  # Tamaño mínimo del problema
    d = 20               # Features
    k = 5                # Número de clusters
    seed = 42            # Semilla para reproducibilidad
    
    # 1. Generar datos distribuidos (cada proceso genera su porción)
    data = generate_distributed_data(n_total, d, k, seed)
    
    if rank == 0:
        print(f"K-means Distribuido iniciado")
        print(f"Datos: n_total={n_total}, d={d}, k={k}")
        print(f"Procesos MPI: {size}")
    
    # 2. Inicializar centroides (solo root)
    centroids = np.zeros((k, d), dtype=np.float64)
    if rank == 0:
        # Selección aleatoria de centroides iniciales
        np.random.seed(seed)
        indices = np.random.choice(data.shape[0], k, replace=False)
        centroids = data[indices].copy()
    
    # 3. Broadcast de centroides usando comm.Bcast(centroids, root=0)
    comm.Bcast(centroids, root=0)
    
    # 4. Calcular distancias
    distances = compute_distances(data, centroids)
    
    # 5. Asignar labels
    labels = assign_labels(distances)
    
    # 6. Calcular sumas locales
    local_sums, local_counts = compute_local_sums(data, labels, k)
    
    # 7. Reducción global usando comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
    global_sums = np.zeros((k, d), dtype=np.float64)
    global_counts = np.zeros(k, dtype=np.int64)
    
    comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
    comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
    
    # 8. Actualizar centroides (promedio de puntos por cluster)
    new_centroids = np.zeros((k, d), dtype=np.float64)
    for i in range(k):
        if global_counts[i] > 0:
            new_centroids[i] = global_sums[i] / global_counts[i]
    
    if rank == 0:
        print(f"\nResultados:")
        print(f"  Puntos por cluster: {global_counts}")
        print(f"  Total de puntos: {global_counts.sum()}")
        print(f"K-means completado exitosamente!")


if __name__ == "__main__":
    kmeans_distributed()
