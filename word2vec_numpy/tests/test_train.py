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
    
    def test_train_loop(self, mock_init, mock_gen_pairs, mock_build_table, 
                        mock_sample_neg, mock_forward, mock_backward):
        raise NotImplementedError("Implementar test para train()")
if __name__ == '__main__':
    unittest.main()