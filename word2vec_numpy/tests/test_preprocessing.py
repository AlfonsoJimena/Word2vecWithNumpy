################################################################################ PRUEBAS (ELIMINAR PARA VERSIóN FINAL) ####################################################################################################
if __name__ == "__main__":
    # Bloque de prueba rápida: te permite ejecutar
    #   python word2vec/preprocessing.py
    # para comprobar que estas funciones hacen lo que esperas, con un texto de juguete.
    texto_prueba = "El gato duerme. El perro duerme. El gato y el perro son amigos."
    tokens = tokenize(texto_prueba)
    print("Tokens:", tokens)

    word2idx, idx2word, freqs = build_vocab(tokens, min_count=1)
    print("Vocabulario:", word2idx)
    print("Frecuencias:", freqs)

    tokens_sub = subsample(tokens, freqs)
    print("Tras subsampling:", tokens_sub)

##########################################################################################################################################################################################################################