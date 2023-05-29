import cv2

captura = cv2.VideoCapture(0)

# Clasificador de rostros
clasificador_rostros = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Clasificador de sonrisas
clasificador_sonrisas = cv2.CascadeClassifier('haarcascade_smile.xml')

while True:
    _, fotograma = captura.read()
    grises = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)

    # Detección de rostros
    rostros = clasificador_rostros.detectMultiScale(grises, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in rostros:
        cv2.rectangle(fotograma, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Región de interés para la detección de sonrisas
        roi_gray = grises[y:y+h, x:x+w]
        roi_color = fotograma[y:y+h, x:x+w]

        # Detección de sonrisas dentro de los rostros
        sonrisas = clasificador_sonrisas.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))

        for (sx, sy, sw, sh) in sonrisas:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
            cv2.putText(fotograma, 'Sonrisa', (x+sx, y+sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Detección de Sonrisas', fotograma)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()
