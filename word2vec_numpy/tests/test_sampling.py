################################################################################ PRUEBAS (ELIMINAR PARA VERSIóN FINAL) ####################################################################################################
if __name__ == "__main__":
    # Prueba rápida con datos de juguete (sin depender de preprocessing.py todavía)
    fake_freqs = {"el": 100, "gato": 10, "duerme": 8, "perro": 9, "sofa": 3}
    fake_word2idx = {w: i for i, w in enumerate(fake_freqs)}

    fake_tokens = [fake_word2idx[w] for w in ["el", "gato", "duerme", "en", "el", "sofa"] if w in fake_word2idx]
    pairs = generate_cbow_pairs(fake_tokens, window_size=2)
    print("Pares (contexto, centro):", pairs)

    table = build_negative_sampling_table(fake_freqs, fake_word2idx, table_size=1000)
    negs = sample_negatives(table, k=3, exclude_idx=0)
    print("Negativos muestreados:", negs)
##########################################################################################################################################################################################################################