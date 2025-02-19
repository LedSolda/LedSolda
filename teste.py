import cv2

# Inicializa a captura de vídeo com a webcam (0 é o índice da câmera)
webcam = cv2.VideoCapture(2)

# Verifica se a webcam foi aberta corretamente
if not webcam.isOpened():
    print("Erro: A câmera não pôde ser aberta.")
else:
    print("Sucesso: Câmera conectada.")

    while True:
        # Captura frame a frame
        validacao, frame = webcam.read()

        # Se o frame foi lido corretamente
        if validacao:
            # Exibe o frame na janela
            cv2.imshow("Vídeo da Webcam", frame)
        else:
            print("Erro: Não foi possível ler o frame.")
            break

        # Espera 5 ms e verifica se a tecla ESC foi pressionada
        key = cv2.waitKey(5)
        if key == 27:  # ESC
            break

    # Libera a captura da webcam e fecha todas as janelas abertas
    webcam.release()
    cv2.destroyAllWindows()
