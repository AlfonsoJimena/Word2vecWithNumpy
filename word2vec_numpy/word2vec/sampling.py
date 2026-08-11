"""
sampling.py
-----------
Generación de los pares de entrenamiento para CBOW, y el negative sampling.

"""

import numpy as np


def generate_cbow_pairs(token_indices: list[int], window_size: int = 2) -> list[tuple[list[int], int]]:
    """
    Recorre la secuencia de tokens y genera pares (contexto, centro) para CBOW.

    Paper usado como referencia: https://arxiv.org/pdf/1301.3781.pdf
        Efficient Estimation of Word Representations in Vector Space, Mikolov et al., 2013.
        3.1 Continuous Bag-of-Words Model

    Args:
        token_indices: lista de enteros, cada uno el índice de vocabulario
                        de un token.
        window_size: cuántas palabras coger a cada lado del centro.

    Returns:
        Lista de tuplas (contexto, centro), donde:
          - contexto es una lista de ints (longitud variable en los bordes del texto)
          - centro es un int
    """

    resul = []

    for i in range(len(token_indices)):
        centro = token_indices[i]
        
        inicio_izq = max(0, i - window_size)      
        fin_der = min(len(token_indices), i + window_size + 1)   
        
        contexto = token_indices[inicio_izq : i] + token_indices[i+1 : fin_der]
        
        if contexto:
            resul.append((contexto, centro))

    return resul


def build_negative_sampling_table(word_freqs: dict, word2idx: dict,
                                    table_size: int = 1_000_000, power: float = 0.75) -> np.ndarray:
    """
    Precalcula una tabla grande de índices de palabras, donde cada palabra
    aparece un número de veces proporcional a freq(palabra)^power.

    Elevar a 0.75 (en vez de usar la frecuencia directa) es la técnica de 
    suavizado empírica introducida para el Negative Sampling en Word2Vec. 
    Esto aumenta la probabilidad de extraer palabras extremadamente raras y 
    penaliza a las palabras muy frecuentes para que no dominen el muestreo.

    Paper usado como referencia: https://arxiv.org/pdf/1310.4546.pdf
            Distributed Representations of Words and Phrases and their Compositionality, Mikolov et al., 2013.
            2.2 Negative Sampling

    Args:
        word_freqs: dict {palabra: frecuencia_absoluta}
        word2idx: dict {palabra: indice}
        table_size: tamaño de la tabla precomputada.
        power: exponente de suavizado (0.75 por defecto, según el paper original).

    Returns:
        np.ndarray de shape (table_size,) con índices de palabras (dtype=int).
    """

    words = list(word_freqs.keys())
    index = []
    freqs = []

    for word in words:
        index.append(word2idx[word])
        freqs.append(word_freqs[word])
    
    freqs = np.array(freqs)
    freqs_suavizadas = freqs ** power

    probs = freqs_suavizadas / freqs_suavizadas.sum()

    tabla = np.random.choice(index, size=table_size, p=probs)

    return tabla


def sample_negatives(table: np.ndarray, k: int, exclude_idx: int) -> np.ndarray:
    """
    Muestrea k índices negativos de la tabla precomputada, evitando que
    coincidan con la palabra positiva (exclude_idx).

    Args:
        table: la tabla devuelta por build_negative_sampling_table().
        k: cuántos negativos se necesitan (típicamente 5-15 para corpus pequeños).
        exclude_idx: índice de la palabra centro real, para no muestrearla por error.

    Returns:
        np.ndarray de shape (k,) con índices de palabras negativas.
    """

    negatives = np.random.choice(table, size=k)

    while exclude_idx in negatives:
        mask = (negatives == exclude_idx)
        negatives[mask] = np.random.choice(table, size=mask.sum())

    return negatives

