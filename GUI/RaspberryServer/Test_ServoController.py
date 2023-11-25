import RPi.GPIO as GPIO
import time

class ServoController:
    def __init__(self):
        self.servo_pin1 = 2
        self.servo_pin2 = 3
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.servo_pin1, GPIO.OUT)
        GPIO.setup(self.servo_pin2, GPIO.OUT)

        self.pwm1 = GPIO.PWM(self.servo_pin1, 50)
        self.pwm2 = GPIO.PWM(self.servo_pin2, 50)

    def set_angle1(self, angle):
        duty = angle / 18 + 2
        GPIO.output(self.servo_pin1, True)
        self.pwm1.ChangeDutyCycle(duty)
        time.sleep(1)
        GPIO.output(self.servo_pin1, False)
        self.pwm1.ChangeDutyCycle(0)
    def set_angle2(self, angle):
        duty = angle / 18 + 2
        GPIO.output(self.servo_pin2, True)
        self.pwm2.ChangeDutyCycle(duty)       
        time.sleep(1)
        GPIO.output(self.servo_pin2, False)
        self.pwm2.ChangeDutyCycle(0)

servo=ServoController()
servo.set_angle1(90)
servo.set_angle2(90)
