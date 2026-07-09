"""
sampling.py
-----------
Generación de los pares de entrenamiento para CBOW, y el muestreo de
negativos (negative sampling).

En CBOW, cada ejemplo de entrenamiento es:
    ([contexto_1, contexto_2, ..., contexto_n], centro)
es decir: varias palabras de contexto que deben predecir UNA palabra centro.
(Al contrario que en skip-gram, donde era 1 centro -> N pares con cada contexto).
"""

import numpy as np


def generate_cbow_pairs(token_indices: list[int], window_size: int = 2) -> list[tuple[list[int], int]]:
    """
    Recorre la secuencia de tokens (ya convertidos a índices de vocabulario)
    y genera pares (contexto, centro) para CBOW.

    Para cada posición i en la secuencia:
        centro = token_indices[i]
        contexto = token_indices[i-window_size : i] + token_indices[i+1 : i+window_size+1]
        (con cuidado en los bordes del texto, para no salirte del array)

    Args:
        token_indices: lista de enteros, cada uno el índice de vocabulario
                        de un token (ya deberías haber pasado tokens -> word2idx antes).
        window_size: cuántas palabras coger a cada lado del centro.

    Returns:
        Lista de tuplas (contexto, centro), donde:
          - contexto es una lista de ints (longitud variable en los bordes del texto)
          - centro es un int

        Ej: [([5, 8, 12, 3], 9), ([8, 9, 3, 44], 12), ...]
    """
    # TODO: iterar sobre cada posición i de token_indices
    # TODO: para cada i, construir la ventana de contexto respetando los límites
    #       (cuidado: si i-window_size < 0 o i+window_size >= len(...), recorta)
    # TODO: opcional pero recomendado -> descartar pares donde el contexto quede vacío
    #       (puede pasar en textos muy cortos o en los extremos)
    raise NotImplementedError


def build_negative_sampling_table(word_freqs: dict, word2idx: dict,
                                    table_size: int = 1_000_000, power: float = 0.75) -> np.ndarray:
    """
    Precalcula una tabla grande de índices de palabras, donde cada palabra
    aparece un número de veces proporcional a freq(palabra)^power.

    Elevar a 0.75 (en vez de usar la frecuencia directa) es la receta del
    paper original: suaviza la distribución para que las palabras rarísimas
    tengan algo más de probabilidad, y las poquísimas palabras ultra-frecuentes
    no dominen tanto el muestreo.

    Args:
        word_freqs: dict {palabra: frecuencia_absoluta}
        word2idx: dict {palabra: indice}
        table_size: tamaño de la tabla precomputada (más grande = muestreo más fino,
                    pero más memoria).
        power: exponente de suavizado, 0.75 en el paper original.

    Returns:
        np.ndarray de shape (table_size,) con índices de palabras (dtype=int),
        listo para hacer np.random.choice(table) súper rápido en el bucle de entrenamiento.
    """
    # TODO: para cada palabra, calcular freq^power
    # TODO: normalizar para que sumen 1 (son "probabilidades")
    # TODO: repartir table_size huecos proporcionalmente a esas probabilidades
    #       (pista: np.random.choice(list(word2idx.values()), size=table_size, p=probs)
    #        es más simple que construir la tabla "a mano", y es igual de válido)
    raise NotImplementedError


def sample_negatives(table: np.ndarray, k: int, exclude_idx: int) -> np.ndarray:
    """
    Muestrea k índices negativos de la tabla precomputada, evitando que
    coincidan con la palabra positiva (exclude_idx).

    Args:
        table: la tabla devuelta por build_negative_sampling_table().
        k: cuántos negativos quieres (típicamente 5-15 para corpus pequeños).
        exclude_idx: índice de la palabra centro real, para no muestrearla por error.

    Returns:
        np.ndarray de shape (k,) con índices de palabras negativas.
    """
    # TODO: np.random.choice(table, size=k) y, si algún resultado == exclude_idx,
    #       volver a muestrear ese hueco (o simplemente ignorarlo, con tablas grandes
    #       la probabilidad de colisión es baja y no es grave para este ejercicio)
    raise NotImplementedError


if __name__ == "__main__":
    # Prueba rápida con datos de juguete (sin depender de preprocessing.py todavía)
    fake_freqs = {"el": 100, "gato": 10, "duerme": 8, "perro": 9, "sofa": 3}
    fake_word2idx = {w: i for i, w in enumerate(fake_freqs)}

    fake_tokens = [fake_word2idx[w] for w in ["el", "gato", "duerme", "en", "el", "sofa"] if w in fake_word2idx]
    pairs = generate_cbow_pairs(fake_tokens, window_size=2)
    print("Pares (contexto, centro):", pairs)

    table = build_negative_sampling_table(fake_freqs, fake_word2idx, table_size=1000)
    negs = sample_negatives(table, k=3, exclude_idx=0)
    print("Negativos muestreados:", negs)
