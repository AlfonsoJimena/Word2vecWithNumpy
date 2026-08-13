"""
train.py
--------
Este archivo actúa como el motor central del modelo CBOW, integrando y coordinando
los componentes de preprocesamiento de datos y las operaciones matemáticas base.
Implementa el ciclo iterativo de optimización sobre el corpus de texto.
"""

import numpy as np
from tqdm import tqdm 

from word2vec.sampling import generate_cbow_pairs, build_negative_sampling_table, sample_negatives
from word2vec.model import init_weights, forward_cbow, backward_cbow


def sgd_update(W_in: np.ndarray, W_out: np.ndarray, context_idxs: list[int],
               center_idx: int, negative_idxs: np.ndarray, grads: dict, lr: float) -> None:
    """
    Aplica un paso de descenso de gradiente (SGD) actualizando las filas
    correspondientes de W_in y W_out IN PLACE (por eso no devuelve nada).

    Args:
        W_in, W_out: las matrices de embeddings (se modifican directamente).
        context_idxs: índices de las palabras de contexto de este ejemplo.
        center_idx: índice de la palabra centro (positiva).
        negative_idxs: índices de las palabras negativas.
        grads: dict devuelto por backward_cbow().
        lr: learning rate.
    """
    
    np.add.at(W_in, context_idxs, -lr * grads["grad_context"])
    W_out[center_idx] -= lr * grads["grad_u_o"]
    np.add.at(W_out, negative_idxs, -lr * grads["grad_u_neg"])

 

def train(token_indices: list[int], vocab_size: int, word_freqs: dict, word2idx: dict,
          embedding_dim: int = 50, window_size: int = 2, k: int = 5,
          lr: float = 0.025, epochs: int = 5, seed: int | None = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Bucle de entrenamiento completo.

    Args:
        token_indices: corpus completo ya convertido a índices de vocabulario.
        vocab_size: V.
        word_freqs, word2idx: para construir la tabla de negative sampling.
        embedding_dim: D.
        window_size: tamaño de ventana de contexto para CBOW.
        k: número de negativos por ejemplo positivo.
        lr: learning rate.
        epochs: número de pasadas completas sobre el corpus.
        seed: semilla para reproducibilidad.

    Returns:
        W_in, W_out entrenados.
    """
    if seed is not None:
        np.random.seed(seed)

    W_in, W_out = init_weights(vocab_size, embedding_dim)

    pairs = generate_cbow_pairs(token_indices, window_size)

    if len(pairs) == 0:
        raise ValueError("¡No se ha generado ningún par de entrenamiento! Revisa el tamaño de tu corpus o baja el MIN_COUNT.")
    
    neg_table = build_negative_sampling_table(word_freqs, word2idx)

    for epoch in range(epochs):
        np.random.shuffle(pairs)  
                                
        total_loss = 0.0

        for context_idxs, center_idx in tqdm(pairs, desc=f"epoch {epoch + 1}/{epochs}"):
            negatives = sample_negatives(neg_table, k, exclude_idx=center_idx)
            loss, cache = forward_cbow(context_idxs, center_idx, negatives, W_in, W_out)
            grads = backward_cbow(cache)
            sgd_update(W_in, W_out, context_idxs, center_idx, negatives, grads, lr)
            total_loss += loss

        avg_loss = total_loss / len(pairs)  
        print(f"Epoch {epoch + 1}/{epochs} - loss media: {avg_loss:.4f}")

    return W_in, W_out
