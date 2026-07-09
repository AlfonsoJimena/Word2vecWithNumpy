"""
preprocessing.py
-----------------
Todo lo relacionado con convertir texto crudo en algo que el modelo pueda usar:
tokenizar, construir vocabulario y hacer subsampling de palabras muy frecuentes.

No depende de NumPy en su mayor parte (esto es procesamiento de texto puro),
salvo subsample() donde sí conviene usar np.random para las probabilidades.
"""

import re
from collections import Counter
import numpy as np


def tokenize(text: str) -> list[str]:
    """
    Convierte un texto crudo en una lista de tokens (palabras).

    Args:
        text: el contenido completo del corpus como un único string.

    Returns:
        Lista de tokens
    """
    tokens = re.findall(r"[a-záéíóúñ]+", text.lower())

    return tokens

def build_vocab(tokens: list[str], min_count: int = 5) -> tuple[dict, dict, dict]:
    """
    Construye el vocabulario a partir de la lista de tokens.

    Args:
        tokens: lista de todos los tokens del corpus (puede tener repetidos).
        min_count: descarta palabras que aparecen menos de min_count veces.
                   Esto reduce mucho el tamaño de vocabulario y quita ruido.

    Returns:
        word2idx: dict {palabra: indice}
        idx2word: dict {indice: palabra}
        word_freqs: dict {palabra: frecuencia_absoluta}  (solo de las que sobreviven al filtro)
    """
    word2idx = {}
    idx2word = {}
    word_freqs = {}

    freqs = Counter(tokens)
    for word, count in freqs.items():
        if count>= min_count:
            word_freqs[word] = count

    word_freqs_ordenado = dict(sorted(word_freqs.items(), key=lambda x: x[1], reverse=True))

    for i, word in enumerate(word_freqs_ordenado):
        word2idx[word] = i

    for i, word in enumerate(word_freqs_ordenado):
        idx2word[i] = word

    return word2idx, idx2word, word_freqs


def subsample(tokens: list[str], word_freqs: dict, threshold: float = 1e-5) -> list[str]:
    """
    Elimina probabilísticamente palabras muy frecuentes (the, de, y, el...)
    siguiendo la fórmula del paper de Mikolov:

        P(descartar w) = 1 - sqrt(threshold / freq_relativa(w))

    Esto mejora mucho la calidad de los embeddings porque evita que el
    entrenamiento se sature de palabras poco informativas.

    Args:
        tokens: lista de tokens original (ya filtrada a vocabulario válido).
        word_freqs: dict {palabra: frecuencia_absoluta} de build_vocab().
        threshold: umbral típico 1e-5 (cuanto más bajo, más agresivo el filtrado).

    Returns:
        Nueva lista de tokens, más corta, con las palabras muy frecuentes
        aparecidas menos veces (o incluso eliminadas del todo en algunas posiciones).
    """

    freq_relativa = {}
    p_descartar = {}
    resul = []

    total_tokens = sum(word_freqs.values())

    for word, count in word_freqs.items():
        freq_relativa[word] = count/ total_tokens

    for word, fr in freq_relativa.items():
        p_descartar[word] = 1 - np.sqrt(threshold / fr)

    for word in tokens:
        prob = p_descartar[word]
        if np.random.rand() < prob:
            continue 
        resul.append(word)

    return resul


if __name__ == "__main__":
    # Bloque de prueba rápida: te permite ejecutar
    #   python word2vec/preprocessing.py
    # para comprobar que estas funciones hacen lo que esperas, con un texto de juguete.
    texto_prueba = "El gato duerme. El perro duerme. El gato y el perro son amigos."
    tokens = tokenize(texto_prueba)
    print("Tokens:", tokens)

    word2idx, idx2word, freqs = build_vocab(tokens, min_count=1)
    print("Vocabulario:", word2idx)
    print("Frecuencias:", freqs)

    tokens_sub = subsample(tokens, freqs)
    print("Tras subsampling:", tokens_sub)