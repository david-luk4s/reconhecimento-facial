import cv2
from datetime import datetime

# Caminho haarcascade
detectorFace = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
detectorOlho = cv2.CascadeClassifier('cascade/haarcascade-eye.xml')

# Instanciado Eigen Faces Recognizer
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classifier/classificadorEigen.yml")

height, width = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)
lista = []

while (True):
    conectado, imagem = camera.read()
    imageGray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Deteccao da face baseado no haarcascade
    faceDetect = detectorFace.detectMultiScale(
        imageGray,
        scaleFactor=1.5,
        minSize=(35, 35),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, h, w) in faceDetect:
        # Desenhando retangulo da face
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Detector Olho with face
        region = imagem[y:y+h, x:x+w]
        imageOlhoGray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        olhoDetector = detectorOlho.detectMultiScale(imageOlhoGray)

        for(ox, oy, oh, ow) in olhoDetector:

            # Desenhando retangulo do olho da face detectada
            cv2.rectangle(region, (ox, oy), (ox+ow, oy+oh), (0, 0, 255), 2)
            image = cv2.resize(imageGray[y:y+h, x:x+w], (width, height))

            # Fazendo comparacao da imagem detectada
            id, confianca = reconhecedor.predict(image)

            if id == 1: name = 'David Lucas'
            else: name = 'Nao identificado'

            # Escrevendo texto no frame
            cv2.putText(imagem, name, (x, y + (h + 24)), font, 1, (0, 255, 0))
            cv2.putText(imagem, str(confianca), (x, y + (h + 43)), font, 1, (0, 0, 255))

    # Mostrando frame
    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'): break

camera.release()
cv2.destroyAllWindows()
