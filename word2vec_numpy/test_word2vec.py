import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Importación correcta desde tu módulo
from word2vec.train import sgd_update, train

class TestWord2VecTraining(unittest.TestCase):

    def test_sgd_update_basic(self):
        """Verifica que los pesos se actualizan correctamente según los gradientes."""
        # Configuración inicial
        W_in = np.zeros((5, 2))
        W_out = np.zeros((5, 2))
        
        context_idxs = [0, 1]
        center_idx = 2
        # Ponemos el índice 3 dos veces para comprobar que np.add.at suma correctamente ambos gradientes
        negative_idxs = np.array([3, 3, 4]) 
        
        grads = {
            "grad_context": np.array([0.1, 0.1]), 
            "grad_u_o": np.array([0.5, 0.5]),
            "grad_u_neg": np.array([
                [0.2, 0.2],  # Gradiente para el primer 3
                [0.2, 0.2],  # Gradiente para el segundo 3
                [0.3, 0.3]   # Gradiente para el 4
            ])
        }
        lr = 0.1

        # Ejecución
        sgd_update(W_in, W_out, context_idxs, center_idx, negative_idxs, grads, lr)

        # Verificaciones para W_in
        # W_in[0] y W_in[1] deberían ser -0.1 * [0.1, 0.1] = [-0.01, -0.01]
        np.testing.assert_array_almost_equal(W_in[0], [-0.01, -0.01])
        np.testing.assert_array_almost_equal(W_in[1], [-0.01, -0.01])
        # El resto de W_in no debería cambiar
        np.testing.assert_array_almost_equal(W_in[2], [0.0, 0.0])

        # Verificaciones para W_out
        # W_out[2] (center) debería ser -0.1 * [0.5, 0.5] = [-0.05, -0.05]
        np.testing.assert_array_almost_equal(W_out[2], [-0.05, -0.05])
        
        # W_out[3] (negativo duplicado) debería recibir DOS actualizaciones: -0.1 * 0.2 * 2 = [-0.04, -0.04]
        np.testing.assert_array_almost_equal(W_out[3], [-0.04, -0.04])
        
        # W_out[4] (negativo único) debería ser -0.1 * [0.3, 0.3] = [-0.03, -0.03]
        np.testing.assert_array_almost_equal(W_out[4], [-0.03, -0.03])

    # ACTUALIZADO: Los parches ahora apuntan a word2vec.train
    @patch('word2vec.train.backward_cbow')
    @patch('word2vec.train.forward_cbow')
    @patch('word2vec.train.sample_negatives')
    @patch('word2vec.train.build_negative_sampling_table')
    @patch('word2vec.train.generate_cbow_pairs')
    @patch('word2vec.train.init_weights')
    def test_train_loop(self, mock_init, mock_gen_pairs, mock_build_table, 
                        mock_sample_neg, mock_forward, mock_backward):
        """Verifica que el bucle de entrenamiento llama a las funciones correctas y devuelve las matrices."""
        
        # Configurar los mocks para simular las dependencias
        mock_W_in = np.ones((10, 5))
        mock_W_out = np.ones((10, 5))
        mock_init.return_value = (mock_W_in, mock_W_out)
        
        # Simulamos 2 pares de (contexto, centro)
        mock_gen_pairs.return_value = [([1, 2], 3), ([4, 5], 6)]
        
        mock_build_table.return_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        mock_sample_neg.return_value = np.array([7, 8])
        mock_forward.return_value = (1.5, {"cache_key": "cache_val"}) # (loss, cache)
        
        dummy_grads = {
            "grad_context": np.zeros(5),
            "grad_u_o": np.zeros(5),
            "grad_u_neg": np.zeros((2, 5))
        }
        mock_backward.return_value = dummy_grads

        # Parámetros de prueba
        token_indices = [1, 2, 3, 4, 5, 6]
        vocab_size = 10
        word_freqs = {i: 1 for i in range(10)}
        word2idx = {str(i): i for i in range(10)}
        
        # Ejecución
        W_in_res, W_out_res = train(
            token_indices, vocab_size, word_freqs, word2idx,
            embedding_dim=5, window_size=1, k=2, lr=0.1, epochs=2, seed=42
        )

        # Verificaciones
        # 1. Comprobar que init_weights se llamó correctamente
        mock_init.assert_called_once_with(10, 5)
        
        # 2. Comprobar el número de llamadas en el bucle (2 pares * 2 epochs = 4 iteraciones)
        self.assertEqual(mock_forward.call_count, 4)
        self.assertEqual(mock_backward.call_count, 4)
        
        # 3. Comprobar que devuelve las matrices correctas
        self.assertIs(W_in_res, mock_W_in)
        self.assertIs(W_out_res, mock_W_out)

if __name__ == '__main__':
    unittest.main()