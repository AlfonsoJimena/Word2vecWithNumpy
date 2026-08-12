import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Importación correcta desde tu módulo
from word2vec.train import sgd_update, train

class TestWord2VecTraining(unittest.TestCase):

    def test_sgd_update_basic(self):
        W_in = np.zeros((5, 2))
        W_out = np.zeros((5, 2))

        context_idxs = [0, 2]
        center_idx = 1
        
        negative_idxs = np.array([3, 3]) 

        grads = {
            "grad_context": np.array([1.0, 1.0]),  
            "grad_u_o": np.array([2.0, -2.0]),     
            "grad_u_neg": np.array([[3.0, 3.0],    
                                    [3.0, 3.0]])  
        }
        
        lr = 0.1

        sgd_update(W_in, W_out, context_idxs, center_idx, negative_idxs, grads, lr)

        np.testing.assert_array_almost_equal(W_in[0], [-0.1, -0.1], 
                                             err_msg="Fallo al actualizar W_in (fila 0)")
        np.testing.assert_array_almost_equal(W_in[2], [-0.1, -0.1], 
                                             err_msg="Fallo al actualizar W_in (fila 2)")
        

        np.testing.assert_array_almost_equal(W_in[1], [0.0, 0.0], 
                                             err_msg="Se alteró una fila de W_in que debía quedar intacta")

        np.testing.assert_array_almost_equal(W_out[1], [-0.2, 0.2], 
                                             err_msg="Fallo al actualizar W_out (palabra central)")


        np.testing.assert_array_almost_equal(W_out[3], [-0.6, -0.6], 
                                             err_msg="Fallo crítico: np.add.at no está acumulando los negativos duplicados")
    
    def test_train_end_to_end_integration(self):

        vocab_size = 5
        embedding_dim = 2
        # Simulación de un corpus: "el(0) gato(1) come(2) pescado(3) crudo(4)"
        token_indices = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4] 
        
        # Frecuencias y diccionarios básicos
        word_freqs = {"el": 2, "gato": 2, "come": 2, "pescado": 2, "crudo": 2}
        word2idx = {"el": 0, "gato": 1, "come": 2, "pescado": 3, "crudo": 4}

        # 2. Ejecución: Pasada (1 epoch) por el motor completo
        W_in_final, W_out_final = train(
            token_indices=token_indices,
            vocab_size=vocab_size,
            word_freqs=word_freqs,
            word2idx=word2idx,
            embedding_dim=embedding_dim,
            window_size=1,  # Ventana pequeña
            k=2,            # 2 palabras negativas
            lr=0.01,
            epochs=1,       # Solo 1 iteración para que el test sea rápido
            seed=42
        )

        # 3. Comprobaciones 

        # A) ¿Se han respetado las dimensiones durante todo el proceso?
        self.assertEqual(W_in_final.shape, (vocab_size, embedding_dim), 
                         "Las dimensiones de W_in se corrompieron durante el entrenamiento")
        self.assertEqual(W_out_final.shape, (vocab_size, embedding_dim), 
                         "Las dimensiones de W_out se corrompieron durante el entrenamiento")

        # B) ¿Ha aprendido algo el modelo? 
        out_is_all_zeros = np.all(W_out_final == 0.0)
        self.assertFalse(out_is_all_zeros, 
                         "W_out sigue lleno de ceros. El SGD no está actualizando los pesos.")

    def test_train_empty_corpus_raises_error(self):
        # 1. Preparación: Corpus vacío (simulando que el filtro 
        # MIN_COUNT fue demasiado agresivo y borró todas las palabras)
        token_indices = [] 
        word_freqs = {"el": 1}
        word2idx = {"el": 0}
        
        # 2. Ejecución y Comprobación: Verificación de que salte la red de seguridad
        # El bloque 'with self.assertRaises' captura el error para que el test no falle
        with self.assertRaises(ValueError) as context:
            train(
                token_indices=token_indices,
                vocab_size=1,
                word_freqs=word_freqs,
                word2idx=word2idx,
                epochs=1
            )
        
        self.assertTrue("¡No se ha generado ningún par" in str(context.exception),
                        "No saltó el mensaje de error correcto al quedarse sin pares.")
if __name__ == '__main__':
    unittest.main()