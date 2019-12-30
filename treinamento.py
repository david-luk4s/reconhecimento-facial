import cv2
import os
import numpy as np

# Usando 3 algoritmos de face detect
eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImageWithId():
    '''
        Percorrer diretorio fotos, ler todas imagens com CV2 e organizar
        conjunto de faces com seus respectivos ids
    '''
    pathsImages = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    ids = []

    for pathImage in pathsImages:
        imageFace = cv2.cvtColor(cv2.imread(pathImage), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(pathImage)[-1].split('.')[1])

        ids.append(id)
        faces.append(imageFace)

        cv2.imshow("Face", imageFace)
        cv2.waitKey(10)
    return np.array(ids), faces


ids, faces = getImageWithId()

# Gerando classifier do treinamento
print("Treinando....")
eigenface.train(faces, ids)
eigenface.write('classifier/classificadorEigen.yml')

# fisherface.train(faces, ids)
# fisherface.write('classifier/classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classifier/classificadorLBPH.yml')
print('Treinamento concluído com sucesso!')