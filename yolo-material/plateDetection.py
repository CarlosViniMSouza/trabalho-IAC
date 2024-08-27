import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from paddleocr import PaddleOCR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ocr = PaddleOCR(lang='en', rec_algorithm='CRNN')

#Carregar Modelo
model = YOLO('/content/drive/MyDrive/Models/car_plateDetection2/weights/yolo.pt')

#Filtro
def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image

#Filtro
def adaptive_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    return binary_image

def load_image(image_file):
    img = Image.open(image_file)
    return img

def predict_license_plate(image):
    #Converter imagem para array numpy
    image_np = np.array(image)
    #Detecção com o Modelo YOLOv8
    results = model.predict(image_np)

    #Extrair caixas delimitadoras e recortar as placas
    annotations = []
    for result in results:
        for detection in result.boxes:
            bbox = detection.xyxy[0].tolist()  #Coordenadas das caixas delimitadoras
            confidence = detection.conf[0].item()  #Pontuação da Confiança
            class_id = detection.cls[0].item()  #ID da Classe
            
            #Notação da classe. Nesse caso, id==0 representa car_plate
            if class_id == 0:
                annotations.append({
                    'bbox': bbox,
                    'confidence': confidence
                })
    
    return annotations

#Função para desenhar anotações na imagem
def draw_annotations(image, annotations):
    draw = ImageDraw.Draw(image)
    for ann in annotations:
        bbox = ann['bbox']
        confidence = ann['confidence']
        
        #Desenho da caixa delimitadora
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline='blue', width=3)
        
        #Recorte da área delimitada
        cropped_plate = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

        #Transforma a imagem cropada em array para manipulação
        cropped_plate_arr = np.array(cropped_plate)

        #Pre-processamento da imagem (aplicação de filtros)
        enhanced_image = enhance_contrast(cropped_plate_arr)
        binary_image = adaptive_threshold(enhanced_image)

        #Utilização do OCR
        results = ocr.ocr(binary_image, cls=False, det=False)
        
        if results and results[0][0][1] > 0.6:
            label = results[0][0][0]
        else:
            label = f"car_plate {confidence:.2f}"
        
        ann['ocr_text'] = label

        #Desenha o resultado do OCR na imagem de output
        font = ImageFont.load_default()
        w, h = font.getsize(label)
        draw.rectangle((bbox[0], bbox[1] - h, bbox[0] + w, bbox[1]), fill='blue')
        draw.text((bbox[0], bbox[1] - h), label, fill='white', font=font)
    
    return image

st.title("Reconhecimento de Placas")
st.write("Envie a Imagem para Iniciar a Detecção.")

#Upload da imagem
image_file = st.file_uploader("Enviar Imagem", type=["jpg", "png", "jpeg"])

if image_file is not None:
    #Carregar e mostrar a imagem inserida
    image = load_image(image_file)
    
    #Predição das placas
    annotations = predict_license_plate(image)
    
    #Anotação das imagens
    annotated_image = draw_annotations(image.copy(), annotations)

    #Saída em JSON
    st.subheader("Resultados da Detecção (JSON)")
    st.json(annotations)
    
    #Criação das duas colunas no Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Enviada")
        st.image(image, caption='Imagem Enviada', use_column_width=True)
    
    with col2:
        st.subheader("Imagem Anotada")
        st.image(annotated_image, caption='Imagem com Notação', use_column_width=True)