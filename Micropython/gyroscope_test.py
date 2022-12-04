mport machine
import utime
import sys
import struct
import math



#Now do I2C stuff (We have already trained the model at this point)


# Setup
led_onboard = machine.Pin(25, machine.Pin.OUT)
i2c = machine.I2C(0, scl=machine.Pin(17, pull=machine.Pin.PULL_UP), sda=machine.Pin(16,pull=machine.Pin.PULL_UP), freq=400000)


gyro_address = 106
accel_address = 30
gyro_reg = 32

utime.sleep(1)

#devices = i2c.scan()
# If you can find i2C device, try this:
#if devices:
#    for d in devices:
#        print(hex(d))

def reg_write(i2c, addr, reg, data):
    """
    Write bytes to the specified register.
    """
    
    # Construct message
    msg = bytearray()
    msg.append(data)
    
    # Write out message to register
    i2c.writeto_mem(addr, reg, msg)
    
def reg_read(i2c, addr, reg, nbytes=1):
    """
    Read byte(s) from specified register. If nbytes > 1, read from consecutive
    registers.
    """
    
    # Check to make sure caller is asking for 1 or more bytes
    if nbytes < 1:
        return bytearray()
    
    # Request data from specified register(s) over I2C
    data = i2c.readfrom_mem(addr, reg, nbytes)
    
    int_data = int.from_bytes(data, "big")
    return int_data


data = reg_read(i2c, gyro_address, gyro_reg)
#if (data != bytearray((DEVID,))):
#    print("ERROR: Could not communicate with gyro")
#    sys.exit()
#else:
#    print("This is the data we read \n")
#    print(data)

reg_write(i2c, gyro_address, gyro_reg, 15)
utime.sleep(1)
#i2c_bus.write_byte_data(0x6A, 0x20, 0x0F)

# Now loop do a perament loop and do inference from observations

slice_count = 0

x_acc = []
y_acc = []
z_acc = []


while True:
    

     
    
    #i2c_bus.write_byte_data(0x6A, 0x20, 0x0F)
    
     # LSM330 Gyro address, 0x6A(106)
    # Read data back from 0x28(40), 2 bytes
    # X-Axis Gyro LSB, X-Axis Gyro MSB
    data0 = reg_read(i2c, gyro_address, 0x28)
    data1 = reg_read(i2c, gyro_address, 0x29)

    # Convert the data
    xGyro = int((data1 * 256 + data0))
    if xGyro > 32767 :
        xGyro -= 65536

    # LSM330 Gyro address, 0x6A(106)
    # Read data back from 0x2A(42), 2 bytes
    # Y-Axis Gryo LSB, Y-Axis Gyro MSB
    data0 = reg_read(i2c, gyro_address, 0x2A)
    data1 = reg_read(i2c, gyro_address, 0x2B)

    # Convert the data
    yGyro = data1 * 256 + data0
    if yGyro > 32767 :
        yGyro -= 65536

    # LSM330 Gyro address, 0x6A(106)
    # Read data back from 0x2C(44), 2 bytes
    # Z-Axis Gyro LSB, Z-Axis Gyro MSB
    data0 = reg_read(i2c, gyro_address, 0x2C)
    data1 = reg_read(i2c, gyro_address, 0x2D)

    # Convert the data
    zGyro = data1 * 256 + data0
    if zGyro > 32767 :
        zGyro -= 65536

    # LSM330 Accl address, 0x1D(29)
    # Select control register1, 0x20(32)
    #		0x67(103)	Power ON, Data rate selection = 100 Hz
    #					X, Y, Z-Axis enabled
    #i2c_bus.write_byte_data(30, 0x20, 0x67)

    reg_write(i2c, accel_address, 0x20, 0x67)

    utime.sleep(0.5)

    # LSM330 Accl address, 30(29)
    # Read data back from 0x28(40), 2 bytes
    # X-Axis Accl LSB, X-Axis Accl MSB
    data0 = reg_read(i2c, accel_address, 0x28)
    data1 = reg_read(i2c, accel_address, 0x29)

    # Convert the data
    xAccl = data1 * 256 + data0
    if xAccl > 32767 :
        xAccl -= 65536

    # LSM330 Accl address, 30(29)
    # Read data back from 0x2A(42), 2 bytes
    # Y-Axis Accl LSB, Y-Axis Accl MSB
    data0 = reg_read(i2c, accel_address, 0x2A)
    data1 = reg_read(i2c, accel_address, 0x2B)

    # Convert the data
    yAccl = data1 * 256 + data0
    if yAccl > 32767 :
        yAccl -= 65536

    # LSM330 Accl address, 30(29)
    # Read data back from 0x2C(44), 2 bytes
    # Z-Axis Accl LSB, Z-Axis Accl MSB
    data0 = reg_read(i2c, accel_address, 0x2C)
    data1 = reg_read(i2c, accel_address, 0x2D)

    # Convert the data
    zAccl = data1 * 256 + data0
    if zAccl > 32767 :
        zAccl -= 65536
        
        


    # Output data to screen
    #Use +-250 sensitivty and convert dps to rad/s
    print("X-Axis of Rotation : %f" %((xGyro*math.pi)/(114.285*180)))
    print("Y-Axis of Rotation : %f" %((yGyro*math.pi)/(114.285*180)))
    print("Z-Axis of Rotation : %f" %((zGyro*math.pi)/(114.285*180)))
    
    #Use 0.06 instead of 0.061 (datasheet says "typical" so this fine tune is expected)
    print("Acceleration in X-Axis : %f" %((xAccl*9.80665)/16666.6))
    print("Acceleration in Y-Axis : %f" %((yAccl*9.80665)/16666.6))
    print("Acceleration in Z-Axis : %f" %zAccl)
    print("Acceleration in Z-Axis : %f" %((zAccl*9.80665)/16666.6))
    
    
    xGyro = (xGyro*math.pi)/(114.285*180);
    yGyro = (yGyro*math.pi)/(114.285*180);
    zGyro = (zGyro*math.pi)/(114.285*180);
    
    xAccl = (xAccl*9.80665)/(16666.6);
    yAccl = (yAccl*9.80665)/(16666.6);
    zAccl = (zAccl*9.80665)/(16666.6);
    
    
    
    pitch = 180 * math.atan(xAccl/math.sqrt(yAccl*yAccl + zAccl*zAccl))/(math.pi);
    roll = 180 * math.atan(yAccl/math.sqrt(xAccl*xAccl + zAccl*zAccl))/(math.pi);
    yaw = 180 * math.atan(zAccl/math.sqrt(xAccl*xAccl + zAccl*zAccl))/(math.pi);
    
    
    print("Roll : %f" %roll)
    print("Pitch : %f" %pitch)
    print("Yaw : %f" %yaw)
    
    
    
    


    
    led_onboard.value(1)
    utime.sleep(0.25)
    led_onboard.value(0)
    utime.sleep(0.25)
