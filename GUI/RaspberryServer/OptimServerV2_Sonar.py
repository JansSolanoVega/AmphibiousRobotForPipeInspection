
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
from brping import Ping360
from time import sleep

# Make a new Ping
myPing = Ping360()
#if args.device is not None:
myPing.connect_serial('/dev/ttyUSB0',115200)

if myPing.initialize() is False:
    print("Failed to initialize Ping!")
    exit(1)

# Crear la figura y los ejes
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.set_ylim(0, 10)  # Establecer el límite del radio

def calculateRange(numberOfSamples, samplePeriod, speedOfSound, _samplePeriodTickDuration=25e-9):
    # type: (float, int, float, float) -> float
    """
      Calculate the range based in the duration
     """
    return numberOfSamples * speedOfSound * _samplePeriodTickDuration * samplePeriod / 2

def getSonarData(myPing, angle):
    """
    Transmits the sonar angle and returns the sonar intensities
    Args:
        sensor (Ping360): Sensor class
        angle (int): Gradian Angle
    Returns:
        list: Intensities from 0 - 255
    """
    myPing.transmitAngle(angle)
    data = bytearray(getattr(myPing, '_data'))
    return [k for k in data]
  
def calculateRange(numberOfSamples, samplePeriod, speedOfSound, _samplePeriodTickDuration=25e-9):
    # type: (float, int, float, float) -> float
    """
      Calculate the range based in the duration
     """
    return numberOfSamples * speedOfSound * _samplePeriodTickDuration * samplePeriod / 2


def calculateSamplePeriod(distance, numberOfSamples, speedOfSound, _samplePeriodTickDuration=25e-9):
    # type: (float, int, int, float) -> float
    """
      Calculate the sample period based in the new range
     """
    return 2 * distance / (numberOfSamples * speedOfSound * _samplePeriodTickDuration)


def adjustTransmitDuration(distance, samplePeriod, speedOfSound, _firmwareMinTransmitDuration=5):
    # type: (float, float, int, int) -> float
    """
     @brief Adjust the transmit duration for a specific range
     Per firmware engineer:
     1. Starting point is TxPulse in usec = ((one-way range in metres) * 8000) / (Velocity of sound in metres
     per second)
     2. Then check that TxPulse is wide enough for currently selected sample interval in usec, i.e.,
          if TxPulse < (2.5 * sample interval) then TxPulse = (2.5 * sample interval)
        (transmit duration is microseconds, samplePeriod() is nanoseconds)
     3. Perform limit checking
     Returns:
        float: Transmit duration
     """
    duration = 8000 * distance / speedOfSound
    transmit_duration = max(
        2.5 * getSamplePeriod(samplePeriod) / 1000, duration)
    return max(_firmwareMinTransmitDuration, min(transmitDurationMax(samplePeriod), transmit_duration))


def transmitDurationMax(samplePeriod, _firmwareMaxTransmitDuration=500):
    # type: (float, int) -> float
    """
    @brief The maximum transmit duration that will be applied is limited internally by the
    firmware to prevent damage to the hardware
    The maximum transmit duration is equal to 64 * the sample period in microseconds
    Returns:
        float: The maximum transmit duration possible
    """
    return min(_firmwareMaxTransmitDuration, getSamplePeriod(samplePeriod) * 64e6)

def getSamplePeriod(samplePeriod, _samplePeriodTickDuration=25e-9):
    # type: (float, float) -> float
    """  Sample period in ns """
    return samplePeriod * _samplePeriodTickDuration
    
def SetAngle1(angle):
	angle=int(angle)
	duty = angle / 18 + 2
	GPIO.output(3, True)
	pwm1.ChangeDutyCycle(duty)
	GPIO.output(3, False)

def SetAngle2(angle):
	angle=int(angle)
	duty = angle / 18 + 2
	GPIO.output(2, True)
	pwm2.ChangeDutyCycle(duty)
	GPIO.output(2, False)
	
ser = serial.Serial("/dev/ttyACM0", baudrate=9600)

# Inicializar datos
x = []
y = []

anglerad=0
app=Flask(__name__)
target_linear_vel=0;target_angular_vel=0;angle_x=0;angle_y=0

def generate_frames():
    
    #Create a PiCamera object
    camera = Picamera2()

    # Set the resolution of the camera
    camera.configure(camera.create_video_configuration(main={"format": "XRGB8888", "size": (1920,1080)}))#Wider, less FPS
    camera.configure(camera.create_video_configuration(main={"format": "XRGB8888", "size": (1280,720)}))#Narrow, smooth

    # Start previewing the camera feed
    camera.start()
    while True:
        frame = camera.capture_array()
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_graph():
    
    global anglerad
    global anglesex
    anglesex=0
    while True:
        if anglerad==370:
            anglerad=0
        ax.clear()
        ax.set_ylim(0, 3)  # Establecer el límite del radio
        # Generar datos aleatorios para el radio y el ángulo
        

        data = getSonarData(myPing, anglerad)
        for detectedIntensity in data:
            if detectedIntensity >= threshold:
                detectedIndex = data.index(detectedIntensity)

        radius = calculateRange((1 + detectedIndex), samplePeriod, speedOfSound)

        x.append(anglesex)
        y.append(radius)
        # Graficar el punto en coordenadas polares
        ax.plot(x[-37:], y[-37:])
        
        # Actualizar el gráfico
        fig.canvas.draw()
        
        # Guardar el gráfico como imagen
        fig.savefig('grafica.png', dpi=300)  # Guardar como PNG con una resolución de 300 DPI
        
        # Leer la imagen guardada con OpenCV
        image = cv2.imread('grafica.png')
        image=np.array(image)

        ret,buffer=cv2.imencode('.jpg',image)

        frame=buffer.tobytes()
        anglesex+=np.pi/18
        anglerad+=10
        yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


gain = 0
numberOfSamples = 200 # Number of points
transmitFrequency =  740 # Default frequency
sonarRange = 1 # in m
speedOfSound = 1500 # in m/s

# The function definitions are at the bottom
samplePeriod = calculateSamplePeriod(sonarRange, numberOfSamples, speedOfSound)
transmitDuration = adjustTransmitDuration(sonarRange, samplePeriod, speedOfSound)



# Read and print distance measurements with confidence
threshold=150

@app.route('/')
def index():
    return render_template('index4.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video2')
def video2():
    return Response(generate_graph(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/index2')
def index2():
    return render_template('index4.html')

@app.route('/position')
def position():
    return render_template('positions.html')

@app.route('/open-window')
def new_window():
    return render_template('new_window.html')

time_anterior=0
@app.route('/endpoint', methods=['POST'])
def endpoint():
    global target_linear_vel, target_angular_vel, angle_x, angle_y, time_anterior
    target_linear_vel = request.form['target_linear_vel']
    target_angular_vel = request.form['target_angular_vel']
    angle_x = request.form['angle_x']
    angle_y = request.form['angle_y']
    print(target_linear_vel, target_angular_vel, angle_x, angle_y)
    try:
        informacion = str(target_linear_vel)+','+str(target_angular_vel)+','+str(angle_x)+','+str(angle_y)+'\n'
        if (time.time()-time_anterior)>0.3:
	        ser.write(informacion.encode());time_anterior=time.time()
	        #SetAngle1(angle_x);SetAngle2(angle_y);
                
        #time.sleep(0.2)
        #print('funciono')
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
if __name__=="__main__":
    app.run(host="192.168.1.100",port=8080,debug=True)
