import unittest
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

if __name__ == '__main__':
    unittest.main()