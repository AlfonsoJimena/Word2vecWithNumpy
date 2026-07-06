# Word2vecWithNumpy
Learning to Reason with Small Models (JetBrains)

Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).

## Esquema principal del proyecto: 

```
word2vec-numpy/
├── data/
│   └── corpus.txt          # tu texto crudo
├── word2vec/
│   ├── __init__.py
│   ├── preprocessing.py    # tokenizar, vocabulario, subsampling
│   ├── sampling.py         # pares (centro,contexto) + negative sampling
│   ├── model.py            # inicialización W_in, W_out, forward, gradientes
│   └── train.py            # el bucle de entrenamiento
├── evaluate.py              # similitud coseno, vecinos más cercanos
├── main.py                  # orquesta todo: carga config, llama a train, guarda embeddings
├── requirements.txt          # solo numpy (y quizá tqdm)
└── README.md
```