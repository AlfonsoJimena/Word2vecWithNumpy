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
 
## Arquitectura del proyecto
 
El pipeline completo va desde la extracción del corpus hasta la evaluación de
los embeddings entrenados, pasando por el módulo `word2vec` donde vive el
bucle de entrenamiento:
 
![Diagrama de flujo del proyecto](images/diagrama_flujo.png)
 
## Estructura del proyecto
 
```
Word2VecWithNumpy/
├── images/                       
├── papers/                       # papers usados para el desarrollo
├── word2vec-numpy/
    ├── data/
        └── corpus.txt            
    ├── embeddings/               # se genera al entrenar el modelo
    │   ├── W_in.npy
    │   └── word2idx.json
    ├── tests/                    # se prueban todas las funciones de word2vec/
        ├── __init__.py
        ├── test_model.py
        ├── test_preprocessing.py
        ├── test_sampling.py
        ├── test_train.py
    ├── word2vec/                 # motor
    │   ├── __init__.py
    │   ├── preprocessing.py      # tokenizar, vocabulario, subsampling
    │   ├── sampling.py           # pares (contexto, centro) + negative sampling
    │   ├── model.py              # inicialización, forward, gradientes (CBOW)
    │   └── train.py              # el bucle de entrenamiento (SGD)
    ├── evaluate.py               # similitud coseno, vecinos más cercanos
    ├── main.py                   # orquesta todo el pipeline
├── .gitignore
├── corpusextract.py              # archivo para volcar datos de wikipedia para el modelo
├── LICENSE
├── README.md
└── requirements.txt
```
 
## Bucle de entrenamiento / Training loop
 
Detalle del bucle de entrenamiento (`forward_cbow` → `backward_cbow` →
`sgd_update`) y las formas de los tensores que se pasan entre cada etapa:
 
![Bucle de entrenamiento y esquema de tensores](images/bucle_entrenamiento.png)
 
## Notas de la implementación
 
- Variante: **CBOW** (varias palabras de contexto predicen la palabra centro).
- Optimización: **negative sampling** en vez de softmax completo (más rápido,
  y evita normalizar sobre todo el vocabulario en cada paso).
- Todo el forward/backward está implementado a mano con NumPy — sin autograd.
paper usado el de Mikolov