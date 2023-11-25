# Adapted 2011, Willow Garage, Inc.
import pygame
import time

MAX_LIN_VEL = 1
MAX_ANG_VEL = 6

LIN_VEL_STEP_SIZE = 0.1

ANG_X_STEP_SIZE = 5
ANG_Y_STEP_SIZE = 5

LIGHT_STEP_SIZE = 0.5

MAX_ANG_X = 165
MIN_ANG_X = 5

MAX_ANG_Y = 165
MIN_ANG_Y = 5

MAX_INTENSITY = 100
MIN_INTENSITY = 1

def checkAngleLimitX(angle):
    return constrain(angle, MIN_ANG_X, MAX_ANG_X)

def checkAngleLimitY(angle):
    return constrain(angle, MIN_ANG_Y, MAX_ANG_Y)

def checkLimitIntensity(intensity):
    return constrain(intensity, MIN_INTENSITY, MAX_INTENSITY)

def constrain(input, low, high):
    if input < low:
      input = low
    elif input > high:
      input = high
    else:
      input = input

    return input
def checkLinearLimitVelocity(vel):
    return constrain(vel, -1*MAX_LIN_VEL, MAX_LIN_VEL)

def checkAngularLimitVelocity(vel):
    return constrain(vel, -MAX_ANG_VEL, MAX_ANG_VEL)

def vels(target_linear_vel, target_angular_vel):
    print("currently:\tlinear vel %s\t angular vel %s " % (target_linear_vel,target_angular_vel))

def angs(angle_x, angle_y):
    print("currently:\tangle x %s\t angle y %s " % (angle_x,angle_y))

def checkButtonEvents(gamepad):
    acel_y = gamepad.get_axis(1)*-1
    acel_angular_y = gamepad.get_axis(3)*-1
    directional_buttons = gamepad.get_hat(0)

    x_button_pressed = gamepad.get_button(0)
    o_button_pressed = gamepad.get_button(2)

    button_Y = gamepad.get_button(3)
    button_A = gamepad.get_button(1)

    return x_button_pressed, o_button_pressed, directional_buttons, acel_y, acel_angular_y, button_Y, button_A

def checkLimits(target_linear_vel, target_angular_vel, acel_y, percentageLight, acel_angular_y, angle_x, angle_y, x_button_pressed, o_button_pressed, directional_buttons, button_lights):       
    target_linear_vel = round(checkLinearLimitVelocity(target_linear_vel + LIN_VEL_STEP_SIZE*acel_y),2)
    target_angular_vel = round(checkAngularLimitVelocity(target_angular_vel + LIN_VEL_STEP_SIZE*acel_angular_y),2)

    angle_x = round(checkAngleLimitX(angle_x+ANG_X_STEP_SIZE*directional_buttons[0]),2)
    angle_y = round(checkAngleLimitY(angle_y+ANG_Y_STEP_SIZE*directional_buttons[1]),2)
    
    percentageLight = round(checkLimitIntensity(percentageLight+LIGHT_STEP_SIZE*button_lights),2)
    print(percentageLight)

    target_linear_vel, target_angular_vel, angle_x, angle_y = resetSpeedsAngles(target_linear_vel, target_angular_vel, angle_x, angle_y, x_button_pressed, o_button_pressed)
    return target_linear_vel, target_angular_vel, angle_x, angle_y, percentageLight
    
def resetSpeedsAngles(target_linear_vel, target_angular_vel, angle_x, angle_y, x_button_pressed, o_button_pressed):
    if x_button_pressed :
        target_linear_vel   = 0.0
        target_angular_vel  = 0.0
        
    if o_button_pressed:
        angle_x = 0.0
        angle_y = 0.0
        
    return target_linear_vel, target_angular_vel, angle_x, angle_y

if __name__=="__main__":
    pygame.init()
    gamepad = pygame.joystick.Joystick(0)
    gamepad.init()

    target_linear_vel=0
    target_angular_vel=0

    angle_x = 0
    angle_y = 0
    while 1:
        pygame.event.get()  # clear event queue

        x_button_pressed, o_button_pressed, directional_buttons, acel_y, acel_angular_y = checkButtonEvents(gamepad)

        target_linear_vel, target_angular_vel, angle_x, angle_y = checkLimits(target_linear_vel, target_angular_vel, acel_y, acel_angular_y, angle_x, angle_y, x_button_pressed, o_button_pressed, directional_buttons)
        
        vels(target_linear_vel, target_angular_vel)
        angs(angle_x, angle_y)

        time.sleep(0.1)
        


    
