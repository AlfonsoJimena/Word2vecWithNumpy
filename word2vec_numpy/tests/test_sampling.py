import unittest
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
        raise NotImplementedError("Test not implemented yet.")

    def test_sample_negatives(self):
        raise NotImplementedError("Test not implemented yet.")

if __name__ == '__main__':
    unittest.main()