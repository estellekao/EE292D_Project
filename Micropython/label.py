import machine
import utime
import sys
import struct
import math



#Set up led
led_fall = machine.Pin(15, machine.Pin.OUT)
led_fall.low()

#Set up button
button = machine.Pin(14, machine.Pin.IN, machine.Pin.PULL_DOWN)

# Setup on board
led_onboard = machine.Pin(25, machine.Pin.OUT)



while True:
    
    
    LED = sys.stdin.readline()
    if (LED == "LED_ON"):
        led_fall.high()
        
         for i in range(100):    
            if (button.value() == 1):
                print("FP")
                led_fall.low()
                break
            utime.sleep(0.1)
            
    
    led_fall.low()
    led_onboard.value(1)
    utime.sleep(0.25)
    led_onboard.value(0)
    utime.sleep(0.25)
