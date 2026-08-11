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
    
    v_ctx = W_in[context_idxs].mean(axis=0)  # (D,)
    u_o = W_out[center_idx]  # (D,)
    u_neg = W_out[negative_idxs]  # (k, D)
    score_pos = np.dot(v_ctx, u_o)  # escalar
    scores_neg = np.dot(u_neg, v_ctx)  # (k,)
    loss = -np.log(sigmoid(score_pos)) - np.sum(np.log(sigmoid(-scores_neg) + 1e-10))  # Añadimos un epsilon para evitar log(0)

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

    Recuerda las fórmulas del docstring de arriba del archivo:
        g_pos = sigmoid(score_pos) - 1
        g_neg = sigmoid(scores_neg)          (vector de k elementos)

        dL/d(v_ctx)   = g_pos * u_o + g_neg @ u_neg
        dL/d(u_o)     = g_pos * v_ctx
        dL/d(u_neg_i) = g_neg_i * v_ctx

        dL/d(cada palabra de contexto) = dL/d(v_ctx) / n_context
    """
    # Recuperamos del cache todo lo que forward_cbow ya calculó,
    # para no tener que recalcular nada (evita repetir trabajo).
    v_ctx = cache["v_ctx"]              # (D,) vector de contexto (promedio de W_in)
    u_o = cache["u_o"]                  # (D,) vector de la palabra centro real (W_out)
    u_neg = cache["u_neg"]              # (k, D) vectores de las k palabras negativas (W_out)
    score_pos = cache["score_pos"]      # escalar, v_ctx . u_o (antes de pasar por sigmoide)
    scores_neg = cache["scores_neg"]    # (k,) u_neg @ v_ctx (antes de pasar por sigmoide)

    # n_context: cuántas palabras de contexto se promediaron para formar v_ctx.
    # Lo necesitamos para repartir el gradiente al final (regla de la cadena de una media).
    n_context = len(cache["context_idxs"])

    # g_pos: "cuánto se equivoca" el modelo con la palabra positiva.
    # Si sigmoid(score_pos) ya vale ~1 (predicción perfecta), g_pos ~ 0 -> gradiente casi nulo.
    # Si sigmoid(score_pos) vale ~0 (predicción muy mala), g_pos ~ -1 -> gradiente grande.
    g_pos = sigmoid(score_pos) - 1        # escalar

    # g_neg: "cuánto se equivoca" el modelo con cada palabra negativa.
    # Aquí queremos que sigmoid(scores_neg) tienda a 0 (que el modelo diga "no encaja"),
    # así que cuanto más alto sea sigmoid(scores_neg), mayor es el error y mayor el gradiente.
    g_neg = sigmoid(scores_neg)            # (k,)

    # Gradiente respecto al vector de contexto (v_ctx):
    # combina el "empujón" de la palabra positiva (g_pos * u_o) con la suma de
    # los "empujones" de las k palabras negativas (g_neg @ u_neg hace ese sumatorio
    # de golpe, sin bucle: es lo mismo que sum_i g_neg[i] * u_neg[i]).
    grad_v_ctx = g_pos * u_o + g_neg @ u_neg    # (D,)

    # Gradiente respecto al vector de la palabra centro real (W_out[center_idx]).
    # Le decimos: "muévete un poco hacia v_ctx" (si g_pos es negativo, el signo menos
    # de la actualización SGD hará que u_o se acerque a v_ctx).
    grad_u_o = g_pos * v_ctx                     # (D,)

    # Gradiente respecto a los vectores de las k palabras negativas (W_out[negative_idxs]).
    # np.outer crea, de golpe, una matriz (k, D) donde la fila i es g_neg[i] * v_ctx
    # -- exactamente el gradiente de cada palabra negativa, sin necesitar un bucle for.
    grad_u_neg = np.outer(g_neg, v_ctx)          # (k, D)

    # v_ctx es un PROMEDIO de las palabras de contexto (W_in[context_idxs].mean(axis=0)).
    # Por la regla de la cadena de una media, el gradiente que le llega a v_ctx se
    # reparte A PARTES IGUALES entre todas las palabras que se usaron para promediar.
    # Por eso dividimos por n_context: cada palabra de contexto recibe solo su "cuota".
    grad_context = grad_v_ctx / n_context        # (D,)

    # Empaquetamos los tres gradientes para que sgd_update() los use directamente.
    return {
        "grad_context": grad_context,
        "grad_u_o": grad_u_o,
        "grad_u_neg": grad_u_neg,
    }

