import machine
import utime
import sys
import struct
import math
import uselect
from sys import stdin, exit



#Set up led
led_fall = machine.Pin(15, machine.Pin.OUT)
led_fall.low()

led_train = machine.Pin(17, machine.Pin.OUT)
led_train.low()

led_alive = machine.Pin(16, machine.Pin.OUT)
led_alive.low()

#Set up button
button = machine.Pin(14, machine.Pin.IN, machine.Pin.PULL_DOWN)

# Setup on board\\\\\\\\\\\\\\\\\\\\\\\\\\
led_onboard = machine.Pin(25, machine.Pin.OUT)


def read_serial_input():
    """
    Buffers serial input.
    Writes it to input_line_this_tick when we have a full line.
    Clears input_line_this_tick otherwise.
    """
    # stdin.read() is blocking which means we hang here if we use it. Instead use select to tell us if there's anything available
    # note: select() is deprecated. Replace with Poll() to follow best practises
    select_result = uselect.select([stdin], [], [], 0)
    input_character = ""
    buffered_input = []
    while select_result[0]:
        # there's no easy micropython way to get all the bytes.
        # instead get the minimum there could be and keep checking with select and a while loop
        input_character = stdin.read(1)
        # add to the buffer
        buffered_input.append(input_character)
        # check if there's any input remaining to buffer
        select_result = uselect.select([stdin], [], [], 0)
            
    result = ''.join(buffered_input)
    return result

str_detect_fall = "LED_ON"
str_detect_alive = "ALIVE"
str_detect_train = "TRAIN"

while True:
    

        
    LED = read_serial_input()
    #print("This is LED" + LED)
    
    pressed = False
    
    if (str_detect_fall in LED):
        led_fall.high()
        
        for i in range(100):
        
            if (button.value() == 1):
                print("FP")
                pressed = True
                led_fall.low()
                break
            utime.sleep(0.1)
            
      
    if (str_detect_alive in LED):
        led_alive.high()
        
        
    if (str_detect_train in LED):
         led_train.high()

                
    #if (pressed == False):
     #   print("TP")
        
    led_fall.low()
    led_onboard.value(1)
   
    utime.sleep(0.25)
    led_train.low()
    led_alive.low()
    led_onboard.value(0)
    utime.sleep(0.25)
