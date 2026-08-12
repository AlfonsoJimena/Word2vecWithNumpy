"""
main.py
-------
Orquestación del pipeline completo:
    texto crudo -> preprocesado -> pares CBOW -> entrenamiento -> guardar embeddings

Ejecutar con:
    python main.py
"""

import json
import numpy as np

from word2vec.preprocessing import tokenize, build_vocab, subsample
from word2vec.train import train


# ---- Hiperparámetros ----
CORPUS_PATH = "word2vec_numpy/data/corpus.txt"
MIN_COUNT = 10          # ignora palabras que aparecen menos de N veces
WINDOW_SIZE = 5         # nº de palabras de contexto a cada lado
EMBEDDING_DIM = 100      # dimensión de los vectores (D)
NEGATIVE_SAMPLES = 5    # k, nº de negativos por ejemplo positivo
LEARNING_RATE = 0.025   # tasa de aprendizaje 
EPOCHS = 5              # número de pasadas por el corpus
SEED = 42               # reproducibilidad

OUT_EMBEDDINGS_PATH = "word2vec_numpy/embeddings/W_in.npy"
OUT_VOCAB_PATH = "word2vec_numpy/embeddings/word2idx.json"

def main():
    with open(CORPUS_PATH, encoding="utf-8") as f: 
        text = f.read()

    tokens = tokenize(text)
    word2idx, idx2word, word_freqs = build_vocab(tokens, min_count=MIN_COUNT)
    tokens = [t for t in tokens if t in word2idx]
    tokens = subsample(tokens, word_freqs)
    token_indices = [word2idx[t] for t in tokens if t in word2idx]

    W_in, W_out = train(
        token_indices, vocab_size=len(word2idx), word_freqs=word_freqs,
        word2idx=word2idx, embedding_dim=EMBEDDING_DIM, window_size=WINDOW_SIZE,
        k=NEGATIVE_SAMPLES, lr=LEARNING_RATE, epochs=EPOCHS, seed=SEED,
    )
    
    np.save(OUT_EMBEDDINGS_PATH, W_in)
    with open(OUT_VOCAB_PATH, "w", encoding="utf-8") as f: 
        json.dump(word2idx, f, ensure_ascii=False)
    print("TODO: terminado, embeddings y vocabulario guardados en disco.")


if __name__ == "__main__":
    main()