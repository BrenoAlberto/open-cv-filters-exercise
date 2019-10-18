import keyboard
import numpy as np
import cv2

def reduzRuidoEConverteParaCinza(frame):
    frame = filtroGauss(frame, 3)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def filtroMedia(frame, k):
    return cv2.blur(frame,(k,k))

def filtroLaplace(frame):
    # A profundidade da imagem da saida. Setada para CV_16S para evitar um transbordamento
    ddepth = cv2.CV_16S

    frame = reduzRuidoEConverteParaCinza(frame)

    # Aplica função laplace
    dst = cv2.Laplacian(frame, ddepth, ksize=3)
    
    # Converte de volta para CV_8U
    return cv2.convertScaleAbs(dst)

def filtroGauss(frame, k):
    return cv2.GaussianBlur(frame ,(k,k),0)

def filtroSobel(frame):
    scale = 1
    delta = 0
    # A profundidade da imagem da saida. Setada para CV_16S para evitar um transbordamento
    ddepth = cv2.CV_16S

    frame = reduzRuidoEConverteParaCinza(frame)

    # Calcula as derivativas nas direções x e y utilizando a func Sobel
    grad_x = cv2.Sobel(frame, ddepth, dx=1, dy=0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(frame, ddepth, dx=0, dy=1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    # Converte de volta para CV_8U
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # Retorna uma aproximação do gradiente (obtida pela adição dos dois gradientes direcionais)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

def filtroMediana(frame, k):
    # Aplica o filtro mediana
    return cv2.medianBlur(frame, ksize=k)

def getImshow(frame):
    # Pega a função atual do imshow
    if modo == 'normal':
        return cv2.imshow('frame', frame)
    elif modo == 'sobel':
        return cv2.imshow('frame', filtroSobel(frame))
    elif modo == 'laplace':
        return cv2.imshow('frame', filtroLaplace(frame))
    elif modo == 'media3x3':
        return cv2.imshow('frame', filtroMedia(frame, 3))
    elif modo == 'media5x5':
        return cv2.imshow('frame', filtroMedia(frame, 5))
    elif modo == 'mediana3x3':
        return cv2.imshow('frame', filtroMediana(frame, 3))
    elif modo == 'mediana5x5':
        return cv2.imshow('frame', filtroMediana(frame, 5))
    elif modo == 'gauss3x3':
        return cv2.imshow('frame', filtroGauss(frame, 3))
    elif modo == 'gauss5x5':
        return cv2.imshow('frame', filtroGauss(frame, 5))
    return modo

cap = cv2.VideoCapture(0)

#Define o modo inicial sem filtros
modo = 'normal'

while(True):
    # Capture frame-by-frame
    ret,frame = cap.read()

    # Display the resulting frame
    getImshow(frame)

    if keyboard.is_pressed('a'):
        modo = 'sobel'
    elif keyboard.is_pressed('s'):
        modo = 'laplace'
    elif keyboard.is_pressed('d'):
        modo = 'media3x3'
    elif keyboard.is_pressed('j'):
        modo = 'media5x5'
    elif keyboard.is_pressed('f'):
        modo = 'mediana3x3'
    elif keyboard.is_pressed('k'):
        modo = 'mediana5x5'
    elif keyboard.is_pressed('h'):
        modo = 'gauss3x3'
    elif keyboard.is_pressed('l'):
        modo = 'gauss5x5'
    elif keyboard.is_pressed('n'):
        modo = 'normal'
    elif cv2.waitKey(1) & keyboard.is_pressed('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()