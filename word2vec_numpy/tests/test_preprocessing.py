import unittest
from word2vec_numpy.word2vec.preprocessing import tokenize, build_vocab, subsample

class TestPreprocessing(unittest.TestCase):

    def test_tokenize(self):
        text = 'El gato duerme, tranquilamente, en el sofá.'

        result = tokenize(text)

        self.assertEqual(result, ['el', 'gato', 'duerme', 'tranquilamente', 'en', 'el', 'sofá'])

if __name__ == '__main__':
    unittest.main()