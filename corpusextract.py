"""
descargar_corpus.py
-------------------
Script auxiliar para descargar artículos limpios de la Wikipedia en español
y guardarlos directamente en nuestro corpus.txt.
"""

from datasets import load_dataset
import os

def generar_corpus():
    # Nos aseguramos de que la carpeta de datos existe
    os.makedirs("word2vec_numpy/data", exist_ok=True)
    ruta_salida = "word2vec_numpy/data/corpus.txt"

    print("Conectando con Hugging Face y descargando Wikipedia en español...")
    # puede tardar un poco
    dataset = load_dataset("wikimedia/wikipedia", "20231101.es", split="train")

    # Número de artículos a volcar en el corpus (cuidado con la RAM)
    num_articulos = 10000 

    print(f"Guardando los primeros {num_articulos} artículos en {ruta_salida}...")
    
    with open(ruta_salida, "w", encoding="utf-8") as f:
        for i in range(num_articulos):
            texto = dataset[i]["text"]
            
            # Limpiamos un poco los saltos de línea extra para que sea texto continuo
            texto_limpio = " ".join(texto.split()) 
            f.write(texto_limpio + "\n")

    print("¡Corpus generado con éxito!")

if __name__ == "__main__":
    generar_corpus()