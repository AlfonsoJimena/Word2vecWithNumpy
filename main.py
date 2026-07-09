"""
main.py
-------
Punto de entrada del proyecto. Orquesta el pipeline completo:
    texto crudo -> preprocesado -> pares CBOW -> entrenamiento -> guardar embeddings

Ejecutar con:
    python main.py
"""

import json
import numpy as np

from word2vec.preprocessing import tokenize, build_vocab, subsample
from word2vec.train import train


# ---- Hiperparámetros (juega con estos valores) ----
CORPUS_PATH = "data/corpus.txt"
MIN_COUNT = 5          # ignora palabras que aparecen menos de N veces
WINDOW_SIZE = 2         # nº de palabras de contexto a cada lado
EMBEDDING_DIM = 50      # dimensión de los vectores (D)
NEGATIVE_SAMPLES = 5    # k, nº de negativos por ejemplo positivo
LEARNING_RATE = 0.025
EPOCHS = 5
SEED = 42

OUT_EMBEDDINGS_PATH = "embeddings/W_in.npy"
OUT_VOCAB_PATH = "embeddings/word2idx.json"


def main():
    # 1. Cargar el corpus
    # TODO: with open(CORPUS_PATH, encoding="utf-8") as f: text = f.read()

    # 2. Preprocesar
    # TODO: tokens = tokenize(text)
    # TODO: word2idx, idx2word, word_freqs = build_vocab(tokens, min_count=MIN_COUNT)
    # TODO: tokens = subsample(tokens, word_freqs)   # opcional pero recomendado
    # TODO: token_indices = [word2idx[t] for t in tokens if t in word2idx]

    # 3. Entrenar
    # TODO: W_in, W_out = train(
    #           token_indices, vocab_size=len(word2idx), word_freqs=word_freqs,
    #           word2idx=word2idx, embedding_dim=EMBEDDING_DIM, window_size=WINDOW_SIZE,
    #           k=NEGATIVE_SAMPLES, lr=LEARNING_RATE, epochs=EPOCHS, seed=SEED,
    #       )

    # 4. Guardar resultados
    # TODO: np.save(OUT_EMBEDDINGS_PATH, W_in)
    # TODO: with open(OUT_VOCAB_PATH, "w", encoding="utf-8") as f: json.dump(word2idx, f, ensure_ascii=False)

    print("TODO: pipeline completo pendiente de implementar")


if __name__ == "__main__":
    main()