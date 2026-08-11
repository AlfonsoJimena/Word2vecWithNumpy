import unittest
import numpy as np
from word2vec_numpy.word2vec.sampling import generate_cbow_pairs, build_negative_sampling_table, sample_negatives

class TestSampling(unittest.TestCase):

    def test_generate_cbow_pairs(self):
        indiex = [10, 20, 30, 40, 50]
        size_window = 2

        result = generate_cbow_pairs(indiex, size_window)

        self.assertEqual(result, [
                                    ([20, 30], 10), 
                                    ([10, 30, 40], 20), 
                                    ([10, 20, 40, 50], 30), 
                                    ([20, 30, 50], 40), 
                                    ([30, 40], 50)
                                ])

    def test_build_negative_sampling_table(self):
        word_freqs = {'el':100, 'perro': 10, 'guacamayo': 1}
        word2idx = {'el': 0, 'perro': 1, 'guacamayo': 2}
        table_size = 100000
        power = 0.75


        result = build_negative_sampling_table(word_freqs, word2idx, table_size, power)

        self.assertEqual(len(result), table_size)
        self.assertAlmostEqual((result == 0).sum() / table_size, 0.82, delta=0.1)

    def test_sample_negatives(self):
        table = np.array([0, 1, 2, 3, 4, 5])
        k = 3
        exclude_idx = 2
        
        result = sample_negatives(table, k, exclude_idx)
        
        self.assertEqual(len(result), k)
        self.assertNotIn(exclude_idx, result)
        
if __name__ == '__main__':
    unittest.main()