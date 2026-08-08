import unittest
import numpy as np
from word2vec_numpy.word2vec.preprocessing import tokenize, build_vocab, subsample

class TestPreprocessing(unittest.TestCase):

    def test_tokenize(self):
        text = 'El gato duerme, tranquilamente, en el sofá.'

        result = tokenize(text)

        self.assertEqual(result, ['el', 'gato', 'duerme', 'tranquilamente', 'en', 'el', 'sofá'])

    def test_build_vocab(self):
        tokens = ['el', 'gato', 'duerme', 'tranquilamente', 'en', 'el', 'sofá', 'gato', 'gato', 'el', 'el']

        word2indx, idx2word, word_freqs = build_vocab(tokens, min_count=1)

        self.assertEqual(word2indx, {'el':0, 'gato':1, 'duerme':2, 'tranquilamente':3, 'en':4, 'sofá':5})
        self.assertEqual(idx2word, {0:'el', 1:'gato', 2:'duerme', 3:'tranquilamente', 4:'en', 5:'sofá'})
        self.assertEqual(word_freqs, {'el':4, "gato":3, 'duerme':1, 'tranquilamente':1, 'en':1, 'sofá':1})

    def test_subsample(self):
        np.random.seed(42) # Se fija una semilla para que el test sea reproducible
        tokens = ['el', 'gato', 'duerme', 'tranquilamente', 'en', 'el', 'sofá', 'gato', 'gato', 'el', 'el']
        word_freqs = {'el':4, "gato":3, 'duerme':1, 'tranquilamente':1, 'en':1, 'sofá':1}
        threshold = 0.1 # Umbral alto adaptado a las palabras del corpus para que se descarte 'el' y se mantengan las demás palabras

        result = subsample(tokens, word_freqs, threshold)
        print(result)

        self.assertTrue(result.count('el') < 4)  # él es muy frecuente y debería ser descartado en alguna ocasión.
        self.assertTrue('duerme' in result)  # 'gato' es muy poco frecuente y debería permanecer
        self.assertTrue('tranquilamente' in result)  # 'tranquilamente' es menos frecuente y debería permanecer

if __name__ == '__main__':
    unittest.main()