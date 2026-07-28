"""
model.py
--------
Aquí vive la "física" del modelo: inicialización de pesos, forward pass,
loss y gradientes. Nada de bucles de entrenamiento ni de epochs aquí,
eso va en train.py. Este archivo son funciones matemáticas puras.

Recordatorio de las fórmulas (CBOW con negative sampling):

    v_ctx = promedio de W_in[palabras_de_contexto]        <- vector de contexto
    u_o   = W_out[centro_positivo]
    u_neg = W_out[negativos]                                (k vectores)

    L = -log(sigmoid(u_o . v_ctx)) - sum_i log(sigmoid(-u_neg_i . v_ctx))

    g_pos   = sigmoid(u_o . v_ctx) - 1
    g_neg_i = sigmoid(u_neg_i . v_ctx)

    dL/d(v_ctx)  = g_pos * u_o + sum_i (g_neg_i * u_neg_i)
    dL/d(u_o)    = g_pos * v_ctx
    dL/d(u_neg_i)= g_neg_i * v_ctx

    Y como v_ctx es un promedio de varios vectores de W_in, el gradiente
    que le llega a v_ctx se reparte a PARTES IGUALES entre todas las
    palabras de contexto que se usaron para promediar (regla de la cadena
    de una media: d(mean)/d(cada_elemento) = 1/n).
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

    x_clip = np.clip(x, -30, 30) # Previene overflow en np.exp(). Fuera de este rango el sigmoide ya satura a 0 o 1.

    return 1.0 / (1.0 + np.exp(-x_clip))


def init_weights(vocab_size: int, embedding_dim: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Inicializa las dos matrices de embeddings.

    Args:
        vocab_size: V, tamaño del vocabulario.
        embedding_dim: D, dimensión de cada vector (típico 50-300 para juguetear).
        seed: opcional, para reproducibilidad.

    Returns:
        W_in:  np.ndarray shape (V, D)  -> embeddings "de entrada" (contexto)
        W_out: np.ndarray shape (V, D)  -> embeddings "de salida" (palabra a predecir)
    """
    W_in = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim, size=(vocab_size, embedding_dim))
    W_out = np.zeros((vocab_size, embedding_dim)) # Inicializado con zeros, aunque también podría ser random pequeño.
    return W_in, W_out


def forward_cbow(context_idxs: list[int], center_idx: int, negative_idxs: np.ndarray,
                  W_in: np.ndarray, W_out: np.ndarray) -> tuple[float, dict]:
    """
    Forward pass de un ejemplo CBOW: dado un conjunto de palabras de contexto,
    predice la palabra centro (contra k negativos).

    Args:
        context_idxs: lista de índices de las palabras de contexto (longitud variable).
        center_idx: índice de la palabra centro real (positiva).
        negative_idxs: array de shape (k,) con índices de palabras negativas.
        W_in, W_out: matrices de embeddings actuales.

    Returns:
        loss: float, la pérdida de este ejemplo (para poder monitorizarla en train.py).
        cache: dict con todo lo que backward_cbow() necesitará para no recalcularlo
               (v_ctx, u_o, u_neg, g_pos, g_neg, context_idxs, center_idx, negative_idxs...)

    Pasos:
        1. v_ctx = W_in[context_idxs].mean(axis=0)          # (D,)
        2. u_o   = W_out[center_idx]                         # (D,)
        3. u_neg = W_out[negative_idxs]                      # (k, D)
        4. score_pos = v_ctx . u_o                           # escalar
        5. scores_neg = u_neg @ v_ctx                        # (k,)
        6. loss = -log(sigmoid(score_pos)) - sum(log(sigmoid(-scores_neg)))
           (usa np.clip o suma un epsilon pequeño dentro del log para evitar log(0))
    """
    # TODO: implementar los pasos de arriba
    raise NotImplementedError


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

    Recuerda las fórmulas del docstring de arriba del archivo:
        g_pos = sigmoid(score_pos) - 1
        g_neg = sigmoid(scores_neg)          (vector de k elementos)

        dL/d(v_ctx)   = g_pos * u_o + g_neg @ u_neg
        dL/d(u_o)     = g_pos * v_ctx
        dL/d(u_neg_i) = g_neg_i * v_ctx

        dL/d(cada palabra de contexto) = dL/d(v_ctx) / n_context
    """
    # TODO: recuperar del cache lo necesario (v_ctx, u_o, u_neg, scores, n_context...)
    # TODO: calcular g_pos, g_neg
    # TODO: calcular grad_v_ctx, grad_u_o, grad_u_neg
    # TODO: repartir grad_v_ctx entre las palabras de contexto (dividir por n_context)
    raise NotImplementedError

################################################################################ PRUEBAS (ELIMINAR PARA VERSIóN FINAL) ####################################################################################################
if __name__ == "__main__":
    # Prueba rápida con dimensiones de juguete, para comprobar que las formas
    # (shapes) de todo cuadran antes de meterlo en el bucle de entrenamiento real.
    V, D = 20, 8
    W_in, W_out = init_weights(V, D, seed=42)

    context = [1, 2, 4, 5]
    center = 3
    negatives = np.array([7, 8, 9])

    loss, cache = forward_cbow(context, center, negatives, W_in, W_out)
    print("Loss:", loss)

    grads = backward_cbow(cache)
    print("grad_context shape:", grads["grad_context"].shape)   # esperado: (D,)
    print("grad_u_o shape:", grads["grad_u_o"].shape)             # esperado: (D,)
    print("grad_u_neg shape:", grads["grad_u_neg"].shape)         # esperado: (3, D)
    ##########################################################################################################################################################################################################################