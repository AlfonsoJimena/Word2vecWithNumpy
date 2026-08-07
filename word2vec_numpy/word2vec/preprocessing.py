"""
preprocessing.py
-----------------
Todo lo relacionado con convertir texto crudo en algo que el modelo pueda usar:
tokenizar, construir vocabulario y hacer subsampling de palabras muy frecuentes.
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

def build_vocab(tokens: list[str], min_count: int = 1) -> tuple[dict, dict, dict]:
    """
    Construye el vocabulario a partir de la lista de tokens.
    Filtra las palabras primero y luego asigna los índices.

    Args:
        tokens: lista de todos los tokens del corpus (puede tener repetidos).
        min_count: descarta palabras que aparecen menos de min_count veces.

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

    word_freqs_ordenado = dict(sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)) # Se ordenan las palabras filtradas

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

    Mejora mucho la calidad de los embeddings ya que evita que el
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

