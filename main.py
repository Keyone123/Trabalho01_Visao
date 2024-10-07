import cv2
import numpy as np


def calcular_diferenca_histograma(frame1, frame2):
    # Convertendo para escala de cinza
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculando os histogramas de cada frame
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    # Normalizando os histogramas
    hist1 = cv2.normalize(hist1, hist1)
    hist2 = cv2.normalize(hist2, hist2)

    # Calculando a diferença entre os histogramas
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return diff


def detectar_movimento(frame1, frame2, limiar):
    # Subtração de fundo para detectar áreas de movimento
    diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Aplicando um limiar para detectar áreas de movimento significativas
    _, thres = cv2.threshold(gray_diff, limiar, 255, cv2.THRESH_BINARY)

    # Encontrando contornos nas áreas onde ocorreu movimento
    contornos, _ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhando retângulos em torno das áreas com movimento detectado
    for c in contornos:
        if cv2.contourArea(c) > 500:  # Ignorar pequenos movimentos
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame1


# Captura de vídeo
cap = cv2.VideoCapture(0)

# Inicializando a leitura dos frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # Calculando a diferença de histograma entre dois quadros
    diff_hist = calcular_diferenca_histograma(frame1, frame2)

    # Se a diferença de histograma for significativa, detectamos movimento
    if diff_hist < 0.9:  # Ajustar o valor conforme o teste
        frame_movimento = detectar_movimento(frame1, frame2, limiar=30)
        cv2.imshow('Movimento Detectado', frame_movimento)
    else:
        cv2.imshow('Nenhum Movimento', frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
