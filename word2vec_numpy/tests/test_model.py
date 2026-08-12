import unittest
import numpy as np
from word2vec_numpy.word2vec.model import sigmoid, init_weights, forward_cbow, backward_cbow

class TestModel(unittest.TestCase):

    def test_sigmoid(self):
        x = np.array([0.0, 150.0, -150.0, 2.0])
        
        result = sigmoid(x)
        
        self.assertAlmostEqual(result[0], 0.5, places=4, msg="Debe ser 0.5")
        self.assertAlmostEqual(result[1], 1.0, places=4, msg="Valores altos positivos deben tender a 1.0")
        self.assertAlmostEqual(result[2], 0.0, places=4, msg="Valores altos negativos deben tender a 0.0")
        self.assertAlmostEqual(result[3], 0.8808, places=4, msg="Fallo en el cálculo matemático normal")
        self.assertEqual(result.shape, x.shape, "El array de salida debe tener el mismo tamaño que el de entrada")

    def test_init_weights(self):
        vocab_size = 4
        embedding_dim = 2

        result_W_in, result_W_out = init_weights(vocab_size, embedding_dim, seed=42)

        self.assertEqual(result_W_in.shape, (vocab_size, embedding_dim), "W_in shape incorrecta")
        self.assertEqual(result_W_out.shape, (vocab_size, embedding_dim), "W_out shape incorrecta")
        self.assertTrue(np.all(result_W_out == 0), "W_out debe inicializarse a ceros")
        self.assertTrue(np.all(result_W_in >= -0.5 / embedding_dim), msg="Hay valores por debajo del límite mínimo")
        self.assertTrue(np.all(result_W_in <= 0.5 / embedding_dim), msg="Hay valores por encima del límite máximo")


    def test_forward_cbow(self):
        W_in = np.zeros((5, 2))
        W_in[0] = [1.0, 1.0]   # "el"
        W_in[2] = [3.0, -1.0]  # "come"
        
        W_out = np.zeros((5, 2))
        W_out[1] = [0.5, 0.0]  # "gato" (centro)
        W_out[3] = [-1.0, 0.0] # "piedra" (negativo 1)
        W_out[4] = [0.0, 1.0]  # "nube" (negativo 2)

        context_idxs = [0, 2]
        center_idx = 1
        negative_idxs = np.array([3, 4])
        loss, cache = forward_cbow(context_idxs, center_idx, negative_idxs, W_in, W_out)

        np.testing.assert_array_almost_equal(cache["v_ctx"], [2.0, 0.0], err_msg="Fallo al promediar el contexto")
        np.testing.assert_array_almost_equal(cache["u_o"], [0.5, 0.0], err_msg="Fallo al extraer el vector central") 
        np.testing.assert_array_almost_equal(cache["u_neg"], [[-1.0, 0.0], [0.0, 1.0]], err_msg="Fallo al extraer vectores negativos")   
        self.assertAlmostEqual(cache["score_pos"], 1.0, places=4, msg="Fallo en el producto punto positivo")   
        np.testing.assert_array_almost_equal(cache["scores_neg"], [-2.0, 0.0], err_msg="Fallo en los productos punto negativos")
        expected_loss = 1.1333
        
        self.assertAlmostEqual(loss, expected_loss, places=3, 
                               msg="La función de pérdida matemática es incorrecta")

    def test_backward_cbow(self):
        
        cache = {
            "context_idxs": [0, 2],  
            "v_ctx": np.array([2.0, 0.0]),
            "u_o": np.array([0.5, 0.0]),
            "u_neg": np.array([[-1.0, 0.0], 
                               [ 0.0, 1.0]]),
            "score_pos": 1.0,
            "scores_neg": np.array([-2.0, 0.0])
        }

        gradientes = backward_cbow(cache)

        expected_grad_u_o = np.array([-0.538, 0.0])
        np.testing.assert_array_almost_equal(
            gradientes["grad_u_o"], expected_grad_u_o, decimal=3, 
            err_msg="Fallo en el gradiente de la palabra positiva (u_o)"
        )
        expected_grad_u_neg = np.array([[0.238, 0.0], 
                                        [1.000, 0.0]])
        np.testing.assert_array_almost_equal(
            gradientes["grad_u_neg"], expected_grad_u_neg, decimal=3, 
            err_msg="Fallo en el gradiente de las palabras negativas (u_neg)"
        )
        expected_grad_context = np.array([-0.127, 0.250])
        np.testing.assert_array_almost_equal(
            gradientes["grad_context"], expected_grad_context, decimal=3, 
            err_msg="Fallo al calcular o repartir el gradiente del contexto"
        )

if __name__ == '__main__':
    unittest.main()