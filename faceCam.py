import cv2

captura = cv2.VideoCapture(0)
clasificador_rostros = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while True:
    # Leer el fotograma de la cámara
    _, fotograma = captura.read()

    # Convertir a escala de grises
    grises = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el fotograma
    rostros = clasificador_rostros.detectMultiScale(grises, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar rectángulos alrededor de los rostros detectados
    for (x, y, w, h) in rostros:
        cv2.rectangle(fotograma, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar el fotograma con los rectángulos de los rostros detectados
    cv2.imshow('Reconocimiento de Rostros', fotograma)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
captura.release()
cv2.destroyAllWindows()
