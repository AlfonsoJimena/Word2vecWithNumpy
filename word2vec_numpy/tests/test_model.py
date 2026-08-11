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
        raise NotImplementedError("Test not implemented yet.")

    def test_forward_cbow(self):
        raise NotImplementedError("Test not implemented yet.")

    def test_backward_cbow(self):
        raise NotImplementedError("Test not implemented yet.")

if __name__ == '__main__':
    unittest.main()