"""
train.py
--------
El bucle de entrenamiento propiamente dicho. Junta todo lo de los otros
módulos: genera pares, muestrea negativos, hace forward + backward,
actualiza los pesos con SGD, y va imprimiendo la loss para que puedas
comprobar que baja.
"""

import numpy as np
from tqdm import tqdm  # barra de progreso, opcional pero cómoda

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

    Pista:
        W_in[i] -= lr * grad   para cada i en context_idxs (todas usan el mismo
                                 grad_context, porque el reparto ya se hizo en backward)
        W_out[center_idx] -= lr * grads["grad_u_o"]
        W_out[negative_idxs] -= lr * grads["grad_u_neg"]   (esto es una operación vectorizada,
                                 ojo si hay negativos repetidos: np.add.at es más seguro que -=
                                 directo cuando hay índices duplicados)
    """
    # TODO: implementar las 3 actualizaciones descritas arriba
    raise NotImplementedError


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

    # TODO 1: W_in, W_out = init_weights(...)
    # TODO 2: pairs = generate_cbow_pairs(token_indices, window_size)
    # TODO 3: neg_table = build_negative_sampling_table(word_freqs, word2idx)

    for epoch in range(epochs):
        # TODO 4: barajar 'pairs' en cada epoch (np.random.shuffle no funciona
        #         directo sobre listas de tuplas con distinta longitud interna;
        #         usa random.shuffle de la librería estándar, o baraja los índices)
        total_loss = 0.0

        # for context_idxs, center_idx in tqdm(pairs, desc=f"epoch {epoch}"):
        #     TODO 5: negatives = sample_negatives(neg_table, k, exclude_idx=center_idx)
        #     TODO 6: loss, cache = forward_cbow(context_idxs, center_idx, negatives, W_in, W_out)
        #     TODO 7: grads = backward_cbow(cache)
        #     TODO 8: sgd_update(W_in, W_out, context_idxs, center_idx, negatives, grads, lr)
        #     total_loss += loss

        avg_loss = total_loss / max(len(token_indices), 1)  # ajusta al nº real de pares
        print(f"Epoch {epoch + 1}/{epochs} - loss media: {avg_loss:.4f}")

    return W_in, W_out
