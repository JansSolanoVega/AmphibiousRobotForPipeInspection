import requests
import pygame
from teleoperacion import *
import time

pygame.init()
gamepad = pygame.joystick.Joystick(0)
gamepad.init()

target_linear_vel=0
target_angular_vel=0

angle_x = 0
angle_y = 0

percentageLight=5
while 1:
    pygame.event.get()  # clear event queue

    x_button_pressed, o_button_pressed, directional_buttons, acel_y, acel_angular_y, button_Y, button_A = checkButtonEvents(gamepad)

    
    target_linear_vel, target_angular_vel, angle_x, angle_y, percentageLight = checkLimits(target_linear_vel, target_angular_vel, acel_y, percentageLight, acel_angular_y, angle_x, angle_y, x_button_pressed, o_button_pressed, directional_buttons, button_Y-button_A)

     
    #Light
    
    #vels(target_linear_vel, target_angular_vel)
    #angs(angle_x, angle_y)
    
    data = {'target_linear_vel': target_linear_vel, 'target_angular_vel': target_angular_vel, 'angle_x': angle_x, 'angle_y': angle_y, 'percentage': percentageLight}
    url = 'http://192.168.1.100:8080/endpoint'
    response = requests.post(url, data=data)

    #print(response.text)
