# word2vec-numpy

Implementación de **CBOW con negative sampling** en NumPy puro, sin frameworks
de deep learning (PyTorch/TensorFlow), como ejercicio para entender de primera
mano el bucle de optimización de word2vec: forward pass, loss, gradientes y
actualización de parámetros.

## Instalación

```bash
pip install -r requirements.txt

VersiónPython: 3.12.10
```

## Uso

1. Coloca tu corpus de texto en `data/corpus.txt`.
2. Ajusta hiperparámetros en `main.py` si quieres (dimensión, ventana, epochs...).
3. Entrena:

```bash
python main.py
```

4. Prueba los embeddings resultantes:

```bash
python evaluate.py
```

## Estructura del proyecto

```
word2vec-numpy/
├── data/
│   └── corpus.txt            # tu texto crudo
├── embeddings/               # se genera al entrenar
│   ├── W_in.npy
│   └── word2idx.json
├── word2vec/
│   ├── __init__.py
│   ├── preprocessing.py      # tokenizar, vocabulario, subsampling
│   ├── sampling.py           # pares (contexto, centro) + negative sampling
│   ├── model.py              # inicialización, forward, gradientes (CBOW)
│   └── train.py              # el bucle de entrenamiento (SGD)
├── evaluate.py               # similitud coseno, vecinos más cercanos
└── main.py                   # orquesta todo el pipeline
requirements.txt
README.md
```

## Notas de la implementación

- Variante: **CBOW** (varias palabras de contexto predicen la palabra centro).
- Optimización: **negative sampling** en vez de softmax completo (más rápido,
  y evita normalizar sobre todo el vocabulario en cada paso).
- Todo el forward/backward está implementado a mano con NumPy — sin autograd.

paper usado el de de Mikolov