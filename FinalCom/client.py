import cv2
import time
from teleoperacion import *
import matplotlib.pyplot as plt
import numpy as np
from builtins import input
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
from torchvision import transforms as T
import segmentation_models_pytorch as smp
import requests

# URL del servidor Flask en tu Raspberry Pi
url = 'http://192.168.1.100:8080/download'

output_path = 'videoNuevo.mp4'  # Replace with the desired output path
transform_data = T.Compose([
            T.Resize([224, 224]),
            T.ToTensor()] )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el modelo entrenado
model_read = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=2, activation='softmax')
model_read.load_state_dict(torch.load('unet_Without_TF_resnet34_4kImages_97_Dict.pth'))
# Establecer el modelo en modo de evaluación
model_read.eval()

while 1:
    response = requests.get(url)
    print(response.content)

    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)

        # Open the video file
        video_capture = cv2.VideoCapture(output_path)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        #Crear video
        resolucion = (600, 600)  # Resolución del video (ancho, alto)
        fps = 10  # Cuadros por segundo del video

        nombre_video = "video_salida.avi"  # Nombre del archivo de video de salida
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_salida = cv2.VideoWriter(nombre_video, fourcc, fps, resolucion)
        while True:    
            # Read the next frame
            ret, frame = video_capture.read()
            
            if not ret:
                break

            img = Image.fromarray(frame)
            img = transform_data(img).to(device, dtype=torch.float32).unsqueeze(0)
            
            model_read = model_read.to(device)
            with torch.no_grad():
                a=time.time()
                scores = model_read(img)
                preds = torch.argmax(scores, dim=1).float()
            img = img.cpu()
            preds = preds.cpu().unsqueeze(1)
            
            
            img = img[0].permute(1,2,0).numpy()
            pred = preds[0][0].numpy()
            
            resized_image = cv2.resize(pred, (width, height))
            rgb_img = (cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR))
            
            # Aplica la transparencia a la máscara
            frame = frame.astype(np.float32)/255.0
            dst = frame.copy()
            dst = cv2.addWeighted(frame,0.3,rgb_img,0.7,0, cv2.CV_32F)
            dst=cv2.resize(dst, (600,600))*255
            dst_enteros=cv2.convertScaleAbs(dst)

            #cv2.imshow('Segmented Frame', dst_enteros)
            video_salida.write(dst_enteros)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_salida.release()
        print('Video downloaded successfully!')    

        #ENVIAR VIDEO
        url_envio = 'http://192.168.1.100:8080/upload' 
        with open(nombre_video, 'rb') as archivo:
            # Realiza la solicitud POST al servidor Flask
            respuesta = requests.post(url_envio, files={'videoSeg': archivo})

        # Verifica la respuesta del servidor
        if respuesta.status_code == 200:
            print("Video enviado exitosamente.")
        else:
            print("Error al enviar el video. Código de estado:", respuesta.status_code)
        break
    else:
        print('Failed to download the video.')

    

            
