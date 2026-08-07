################################################################################ PRUEBAS (ELIMINAR PARA VERSIóN FINAL) ####################################################################################################
if __name__ == "__main__":
    # Prueba rápida con dimensiones de juguete, para comprobar que las formas
    # (shapes) de todo cuadran antes de meterlo en el bucle de entrenamiento real.
    V, D = 20, 8
    W_in, W_out = init_weights(V, D, seed=42)

    context = [1, 2, 4, 5]
    center = 3
    negatives = np.array([7, 8, 9])

    loss, cache = forward_cbow(context, center, negatives, W_in, W_out)
    print("Loss:", loss)

    grads = backward_cbow(cache)
    print("grad_context shape:", grads["grad_context"].shape)   # esperado: (D,)
    print("grad_u_o shape:", grads["grad_u_o"].shape)             # esperado: (D,)
    print("grad_u_neg shape:", grads["grad_u_neg"].shape)         # esperado: (3, D)
    ##########################################################################################################################################################################################################################