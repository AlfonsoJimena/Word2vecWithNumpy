"""
evaluate.py
-----------
Cargar los embeddings ya entrenados (.npy + word2idx.json) y comprobar
si tienen sentido: similitud coseno entre palabras, vecinos más cercanos.
"""

import json
import numpy as np


def load_embeddings(npy_path: str = "word2vec_numpy/embeddings/W_in.npy",
                     vocab_path: str = "word2vec_numpy/embeddings/word2idx.json") -> tuple[np.ndarray, dict, dict]:
    """
    Carga la matriz de embeddings y el vocabulario desde disco.

    Returns:
        W_in: np.ndarray shape (V, D)
        word2idx: dict {palabra: indice}
        idx2word: dict {indice: palabra}
    """

    W_in = np.load(npy_path)
    with open(vocab_path, "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    idx2word = {v: k for k, v in word2idx.items()}
    return W_in, word2idx, idx2word


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Similitud coseno entre dos vectores: v1.v2 / (|v1| * |v2|)

    Devuelve un valor entre -1 y 1 (1 = misma dirección, 0 = ortogonales,
    -1 = direcciones opuestas). Cuidado con dividir por cero si algún
    vector tiene norma 0 (añade un epsilon pequeño al denominador).
    """

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)    
    similarity = dot_product / (norm_v1 * norm_v2 + 1e-8)
    return similarity


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
        sin incluir la propia palabra de consulta.

    """

    target_idx = word2idx[word]
    v = W_in[target_idx]                      # Vector de la palabra (D,)
    
    norms = np.linalg.norm(W_in, axis=1)      # Normas de todos los vectores (V,)
    
    sims = (W_in @ v) / (norms * np.linalg.norm(v) + 1e-8)
    
    sorted_idxs = np.argsort(-sims)
    
    results = []
    for idx in sorted_idxs:
        if idx == target_idx:
            continue 
            
        results.append((idx2word[idx], float(sims[idx])))
        
        if len(results) == topn:
            break
            
    return results


if __name__ == "__main__":
    W_in, word2idx, idx2word = load_embeddings()

    palabra_prueba_1 = str(input("1 - Introduce una palabra para ver sus vecinos más cercanos: "))
    palabra_prueba_2 = str(input("2 - Introduce una palabra para ver sus vecinos más cercanos: "))
    palabra_prueba_3 = str(input("3 - Introduce una palabra para ver sus vecinos más cercanos: "))

    palabras_de_prueba = [palabra_prueba_1, palabra_prueba_2, palabra_prueba_3]  
    for w in palabras_de_prueba:
        if w not in word2idx:
            print(f"'{w}' no está en el vocabulario, prueba con otra palabra de tu corpus.")
            continue
        print(f"\nPalabras más parecidas a '{w}':")
        for vecino, sim in most_similar(w, W_in, word2idx, idx2word, topn=5):
            print(f"  {vecino:15s}  {sim:.3f}")
