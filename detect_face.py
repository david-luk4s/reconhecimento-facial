import cv2
import os
import numpy as np

# Caminho Haarcascade
cascPath = 'cascade/haarcascade_frontalface_default.xml'
cascPathOlho = 'cascade/haarcascade-eye.xml'

# Classifier baseado nos haarcascade
facePath = cv2.CascadeClassifier(cascPath)
facePathOlho = cv2.CascadeClassifier(cascPathOlho)
video_capture = cv2.VideoCapture(0)

increment = 1
numMostras = 100
id = input('Digite seu identificador: ')
width, height = 220, 220
print('Capturando as faces...')

# Create directory para salvar on images
os.makedirs('fotos')

while (True):
    conect, image = video_capture.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Qualidade da luz sobre a imagem capturada
    print(np.average(gray))

    # Realizando face detect
    face_detect = facePath.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minSize=(35, 35),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in face_detect:
        # Desenhando retangulo na face detectada
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Realizando deteccao do olho da face
        region = image[y:y + h, x:x + w]
        imageOlhoGray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        face_detect_olho = facePathOlho.detectMultiScale(imageOlhoGray)

        for (ox, oy, ow, oh) in face_detect_olho:
            # Desenhando retangulo nos olhos detectados
            cv2.rectangle(region, (ox, oy), (ox+ow, oy+oh), (0, 0, 255), 2)

            # Salvando imagem com respectivo id para treinamento
            if cv2.waitKey(1) & 0xFF == ord('q'):
                '''
                    Caso queira colocar um delimitador para capturar apenas
                    imagens com uma boa qualidade de luz descomente a linha abaixo
                '''
                # if np.average(gray) > 110:

                face_off = cv2.resize(gray[y:y + h, x:x + w], (width, height))
                cv2.imwrite('fotos/pessoa.' + str(id) + '.' + str(increment) + '.jpg', face_off)

                print('[Foto ' + str(increment) + ' capturada com sucesso] - ', np.average(gray))
                increment += 1

    cv2.imshow('Face', image)
    cv2.waitKey(1)

    if increment > numMostras: break

print('Fotos capturadas com sucesso :)')
video_capture.release()
cv2.destroyAllWindows()