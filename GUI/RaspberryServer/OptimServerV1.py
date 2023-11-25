from flask import Flask,render_template,Response,request
import time
import cv2
import pygame
from teleoperacion import *
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import mpld3
import time
import serial
from picamera2 import Picamera2
import time
import RPi.GPIO as GPIO
from time import sleep
#Agregar
import os
from werkzeug.utils import secure_filename
#---------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(3, GPIO.OUT)
GPIO.setup(2, GPIO.OUT)
pwm1=GPIO.PWM(3, 50)
pwm2=GPIO.PWM(2, 50)
pwm1.start(0)
pwm2.start(0)

def SetAngle1(angle):
	duty = angle / 18 + 2
	GPIO.output(3, True)
	pwm1.ChangeDutyCycle(duty)
	sleep(1)
	GPIO.output(3, False)
	pwm1.ChangeDutyCycle(0)

def SetAngle2(angle):
	duty = angle / 18 + 2
	GPIO.output(2, True)
	pwm2.ChangeDutyCycle(duty)
	sleep(1)
	GPIO.output(2, False)
	pwm2.ChangeDutyCycle(0)
#ser = serial.Serial("/dev/ttyACM0", baudrate=9600)

# Inicializar datos
x = []
y = []

app=Flask(__name__)
#Verificar si son necesarios
#app.config["SECRET_KEY"] ="secretKey"
#app.config["UPLOAD_FOLDER"] = "static\\files"
target_linear_vel=0;target_angular_vel=0;angle_x=0;angle_y=0

def generate_frames():
    
    #Create a PiCamera object
    camera = Picamera2()

    # Set the resolution of the camera
    camera.resolution = (640, 480)

    # Start previewing the camera feed
    camera.start()
    while True:
        # Capture an image and save it
        filename = f"frame.jpg"
        camera.capture_file(filename)

        # Read the captured frame using OpenCV
        frame = cv2.imread(filename)
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames2():
    
    if subido:
        if nombre_archivo[-3:]=="mp4":
                
            transform_data = T.Compose([
                        T.Resize([224, 224]),
                        T.ToTensor()] )
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Cargar el modelo entrenado
            model_read = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=2, activation='softmax')
            model_read.load_state_dict(torch.load('unet_Without_TF_resnet34_4kImages_97_Dict.pth'))

            # Establecer el modelo en modo de evaluación
            model_read.eval()

            # Open the video file
            video_capture = cv2.VideoCapture(nombre_archivo)
            width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            while True:    
                # Read the next frame
                ret, frame = video_capture.read()
                if not ret:
                    break
                
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR)
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
                dst=cv2.resize(dst, (600,600))

                dst = np.array(dst)*255.0
                ret,buffer=cv2.imencode('.jpg',dst)
                frame=buffer.tobytes()
                #cv2.imshow('Segmented Frame', dst)
                #if cv2.waitKey(25) & 0xFF == ord('q'):
                #    break

                yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            while True:  
                dst = cv2.imread("error.png")
                ret,buffer=cv2.imencode('.jpg',dst)
                frame=buffer.tobytes()
                yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index4.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/index2')
def index2():
    return render_template('index4.html')

@app.route('/position')
def position():
    return render_template('positions.html')

@app.route('/open-window')
def new_window():
    return render_template('new_window.html')

@app.route('/open-window2', methods = ['GET', 'POST'])
def new_window2():
    global subido, nombre_archivo
    if request.method == 'POST':
      f = request.files['file']
      nombre_archivo = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
      f.save(nombre_archivo)
      subido=1
      #return 'file uploaded successfully'
    return render_template('NNWindow.html')

@app.route('/endpoint', methods=['POST'])
def endpoint():
    global target_linear_vel, target_angular_vel, angle_x, angle_y
    target_linear_vel = request.form['target_linear_vel']
    target_angular_vel = request.form['target_angular_vel']
    angle_x = request.form['angle_x']
    angle_y = request.form['angle_y']
    print(target_linear_vel, target_angular_vel, angle_x, angle_y)
    try:
        informacion = str(target_linear_vel)+','+str(target_angular_vel)+','+str(angle_x)+','+str(angle_y)+'\n'
        #ser.write(informacion.encode())
        SetAngle1(90) 
        SetAngle2(90)
        time.sleep(0.2)
        print('funciono')
    except:
        pass
    return 'Gracias por enviar los valores'
    # hacer algo con los valores recibidos
    
    #yield f"data: Angle x:  {angle_x}, Angle y:  {angle_y}  ------------- Linear Speed: {target_linear_vel}, Angular Speed: {target_angular_vel}\n\n"
    #return render_template('positions.html')
@app.route('/stream1')
def stream1():
    global target_linear_vel, target_angular_vel, angle_x, angle_y
    def generate():
        while True:
            yield f"data: Angle x:  {angle_x}, Angle y:  {angle_y}  ------------- Linear Speed: {target_linear_vel}, Angular Speed: {target_angular_vel}\n\n"
            time.sleep(0.1)
    return app.response_class(generate(), mimetype='text/event-stream')

@app.route('/data')
def data():
    distance=np.random.uniform(0.8,1.2)
    angle=np.random.uniform(0,3)
    # Generar los datos para la gráfica
    x.append(angle)
    y.append(distance)
    
    # # Crear la gráfica con Matplotlib
    # fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

    #Codigo prueba
    fig=plt.Figure(figsize=(6, 6))
    ax = fig.add_subplot(polar = True)


    ax.plot(x, y)
    
    # Convertir la gráfica a un objeto HTML utilizando mpld3
    html = mpld3.fig_to_html(fig)
    
    # Devolver la respuesta con la gráfica
    return Response(html, mimetype="text/html")

if __name__=="__main__":
    app.run(host="192.168.1.100",port=8080,debug=True)
