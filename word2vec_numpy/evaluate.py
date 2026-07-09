"""
evaluate.py
-----------
Cargar los embeddings ya entrenados (.npy + word2idx.json) y comprobar
si tienen sentido: similitud coseno entre palabras, vecinos más cercanos.

Este script se ejecuta MUCHAS veces mientras iteras, sin tener que
re-entrenar cada vez, por eso está separado del resto.
"""

import json
import numpy as np


def load_embeddings(npy_path: str = "embeddings/W_in.npy",
                     vocab_path: str = "embeddings/word2idx.json") -> tuple[np.ndarray, dict, dict]:
    """
    Carga la matriz de embeddings y el vocabulario desde disco.

    Returns:
        W_in: np.ndarray shape (V, D)
        word2idx: dict {palabra: indice}
        idx2word: dict {indice: palabra}
    """
    # TODO: W_in = np.load(npy_path)
    # TODO: cargar word2idx desde el json
    # TODO: construir idx2word invirtiendo word2idx
    raise NotImplementedError


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Similitud coseno entre dos vectores: v1.v2 / (|v1| * |v2|)

    Devuelve un valor entre -1 y 1 (1 = misma dirección, 0 = ortogonales,
    -1 = direcciones opuestas). Cuidado con dividir por cero si algún
    vector tiene norma 0 (añade un epsilon pequeño al denominador).
    """
    # TODO: implementar
    raise NotImplementedError


def most_similar(word: str, W_in: np.ndarray, word2idx: dict, idx2word: dict, topn: int = 10) -> list[tuple[str, float]]:
    """
    Dada una palabra, devuelve las topn palabras más parecidas del vocabulario
    según similitud coseno de sus vectores en W_in.

    Args:
        word: la palabra de consulta (debe existir en word2idx).
        W_in: matriz de embeddings.
        word2idx, idx2word: vocabulario.
        topn: cuántos vecinos devolver.

    Returns:
        Lista de tuplas (palabra, similitud), ordenada de más a menos parecida,
        SIN incluir la propia palabra de consulta.

    Pista de vectorización (evita un bucle for palabra a palabra):
        v = W_in[word2idx[word]]                      # (D,)
        norms = np.linalg.norm(W_in, axis=1)           # (V,)
        sims = (W_in @ v) / (norms * np.linalg.norm(v) + 1e-8)   # (V,)
        luego np.argsort(-sims) para ordenar de mayor a menor.
    """
    # TODO: implementar de forma vectorizada
    raise NotImplementedError


if __name__ == "__main__":
    W_in, word2idx, idx2word = load_embeddings()

    palabras_de_prueba = ["rey", "gato", "casa"]  # cambia esto por palabras de TU corpus
    for w in palabras_de_prueba:
        if w not in word2idx:
            print(f"'{w}' no está en el vocabulario, prueba con otra palabra de tu corpus.")
            continue
        print(f"\nPalabras más parecidas a '{w}':")
        for vecino, sim in most_similar(w, W_in, word2idx, idx2word, topn=5):
            print(f"  {vecino:15s}  {sim:.3f}")
