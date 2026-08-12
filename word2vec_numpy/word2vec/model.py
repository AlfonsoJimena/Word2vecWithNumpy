"""
model.py
--------
Módulo principal del modelo CBOW (Continuous Bag-of-Words) con Negative Sampling.

Este archivo contiene exclusivamente las operaciones matemáticas fundamentales:
inicialización de matrices de pesos, propagación hacia adelante (forward pass),
cálculo de la función de pérdida (loss) y retropropagación de gradientes.
La lógica iterativa y los bucles de entrenamiento se delegan al módulo `train.py`.
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoide numéricamente estable.

    Args:
        x: np.ndarray 
    
    Returns:
        resul: np.ndarray cuyos elementos se han pasado por un sigmoide
    """

    x_clip = np.clip(x, -30, 30) 

    return 1.0 / (1.0 + np.exp(-x_clip))


def init_weights(vocab_size: int, embedding_dim: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Inicializa las dos matrices de embeddings.

    Args:
        vocab_size: V, tamaño del vocabulario.
        embedding_dim: D, dimensión de cada vector.
        seed: opcional, para reproducibilidad.

    Returns:
        W_in:  np.ndarray shape (V, D)  -> embeddings "de entrada" (contexto)
        W_out: np.ndarray shape (V, D)  -> embeddings "de salida" (palabra a predecir)
    """
    if seed is not None: 
        np.random.seed(seed)

    W_in = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim, size=(vocab_size, embedding_dim))
    W_out = np.zeros((vocab_size, embedding_dim)) 
    return W_in, W_out


def forward_cbow(context_idxs: list[int], center_idx: int, negative_idxs: np.ndarray,
                  W_in: np.ndarray, W_out: np.ndarray) -> tuple[float, dict]:
    """
    Forward pass de un ejemplo CBOW con Negative Sampling.

    Paper usado como referencia: https://arxiv.org/pdf/1411.2738.pdf
        word2vec Parameter Learning Explained, Xin Rong, 2014.
        3.1 Continuous Bag-of-Words Model y 4. Negative Sampling

    Args:
        context_idxs: lista de índices de las palabras de contexto (longitud variable).
        center_idx: índice de la palabra centro real (positiva).
        negative_idxs: array de shape (k,) con índices de palabras negativas.
        W_in, W_out: matrices de embeddings actuales.

    Returns:
        loss: float, la pérdida de este ejemplo (para poder monitorizarla en train.py).
        cache: dict con todo lo que backward_cbow() necesitará para no recalcularlo
               (v_ctx, u_o, u_neg, g_pos, g_neg, context_idxs, center_idx, negative_idxs...)
    """
    
    v_ctx = W_in[context_idxs].mean(axis=0)  
    u_o = W_out[center_idx]  
    u_neg = W_out[negative_idxs]  
    score_pos = np.dot(v_ctx, u_o)  
    scores_neg = np.dot(u_neg, v_ctx) 
    loss = -np.log(sigmoid(score_pos)) - np.sum(np.log(sigmoid(-scores_neg) + 1e-10))  

    return loss, {
        "v_ctx": v_ctx,
        "u_o": u_o,
        "u_neg": u_neg,
        "score_pos": score_pos,
        "scores_neg": scores_neg,
        "context_idxs": context_idxs,
        "center_idx": center_idx,
        "negative_idxs": negative_idxs
    }


def backward_cbow(cache: dict) -> dict:
    """
    Backward pass: calcula los gradientes de la loss respecto a cada
    vector implicado (v_ctx repartido entre las palabras de contexto,
    u_o, y cada u_neg_i).

    Args:
        cache: el dict devuelto por forward_cbow().

    Returns:
        dict con, como mínimo:
          - "grad_context": np.ndarray shape (D,) -> gradiente que le toca A CADA
                             palabra de contexto (ya dividido entre n_context, porque
                             v_ctx es una media)
          - "grad_u_o": np.ndarray shape (D,) -> gradiente para W_out[center_idx]
          - "grad_u_neg": np.ndarray shape (k, D) -> gradiente para W_out[negative_idxs]
    """

    v_ctx = cache["v_ctx"]              
    u_o = cache["u_o"]                  
    u_neg = cache["u_neg"]              
    score_pos = cache["score_pos"]      
    scores_neg = cache["scores_neg"]    

    
    n_context = len(cache["context_idxs"])

    g_pos = sigmoid(score_pos) - 1        # escalar

    g_neg = sigmoid(scores_neg)            # (k,)

    grad_v_ctx = g_pos * u_o + g_neg @ u_neg    # (D,)

    grad_u_o = g_pos * v_ctx                     # (D,)

    grad_u_neg = np.outer(g_neg, v_ctx)          # (k, D)

    grad_context = grad_v_ctx / n_context        # (D,)

    return {
        "grad_context": grad_context,
        "grad_u_o": grad_u_o,
        "grad_u_neg": grad_u_neg,
    }

