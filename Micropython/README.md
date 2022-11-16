# How to get Raspberry Pi Pico Adaptive Fall Detection Running

1. Plug Raspberry Pi Pico into computer
2. Install Micropython UF2 File onto device. See instructions here for more concrete details: https://www.raspberrypi.com/documentation/microcontrollers/micropython.html
3. Install Thonny IDE.  https://thonny.org/ . It is a bit clunky, and hard to add new files to, but has nice serial communication with Pico
4. Optional: Skim this reference https://hackspace.raspberrypi.com/books/micropython-pico. In particular look at the Pinout, the LED, ad pushbutton sections
5. Optional: Skim I2C documentation for Micropython. https://docs.micropython.org/en/latest/library/machine.I2C.html
6. In Thonny, select Micropython (Raspberry Pi Pico) in the bottom left. <img width="1433" alt="image" src="https://user-images.githubusercontent.com/54165966/202307653-5ff72d10-8e9e-4ddf-aca6-2aa2b5859eb6.png">
8. In Thonny, select file "new" and copy and paste the contents of main.py and data.csv. Thonny has encoding "features" that make direct loading difficult so please do this more cumbersome method instead <img width="1433" alt="image" src="https://user-images.githubusercontent.com/54165966/202307840-75c85211-aaac-4211-8070-0475d792d4f0.png">
9. Wire up components. Please use these breadboard pictures for reference. You may change the pin numbers but the code will have to updated. Note: The resistor is a 27 ohm resistor but any reistor in the range of 20-300 ohms should suffice.
![image](https://user-images.githubusercontent.com/54165966/202309305-f7aee928-1f49-4073-b093-266fa2b4da4e.png)
![image](https://user-images.githubusercontent.com/54165966/202309397-973adee7-10da-4a1a-9f32-9af1ae29674f.png)
![image](https://user-images.githubusercontent.com/54165966/202309446-9629e6f1-eeba-40de-90a9-f879f50e5940.png)


10. Press the Green "play" button in Thonny. You should see data printed to the Thonny console
![image](https://user-images.githubusercontent.com/54165966/202308230-dcc73c5f-114b-4deb-a246-9921103f1f72.png)

11. Move accelerometer around and try to get LED to light up. Press the button and see if re-training is initiated. You should see retraining information print to console.

**![image](https://user-images.githubusercontent.com/54165966/202310765-e278b132-2450-464c-88ab-6e156b821b51.png)
**

Retraining stuff you should see:
<img width="1433" alt="image" src="https://user-images.githubusercontent.com/54165966/202310544-36163ae0-2746-4ad6-acb4-c32d0595d57f.png">


## Troubleshooting

### I2C address problems

1. Check the Address jumpers on the LSM 330 accelerometer.
2. Checkout out the I2C documentation and do an i2c.scan() to see the addresses of the devices.

### Memory Allocation problems
1. Reduces the number of lines in the CSV and otherwise remove items we dont need to hold in memory
2. Open issue: We don't have much memory :(


