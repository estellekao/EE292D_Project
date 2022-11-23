# Distributed with a free-will license.
# Use it any way you want, profit or free, provided it fits in the licenses of its associated works.
# LSM330
# This code is designed to work with the LSM330_I2CS I2C Mini Module available from ControlEverything.com.
# https://www.controleverything.com/content/Accelorometer?sku=LSM330_I2CS#tabs-0-product_tabset-2

import smbus
import time

# Get I2C bus
bus = smbus.SMBus(1)

# LSM330 Gyro address, 0x6A(106)
# Select control register1, 0x20(32)
#		0x0F(15)	Power ON, Data rate selection = 95 Hz
#					X, Y, Z-Axis enabled
bus.write_byte_data(0x6A, 0x20, 0x0F)

time.sleep(0.5)

# LSM330 Gyro address, 0x6A(106)
# Read data back from 0x28(40), 2 bytes
# X-Axis Gyro LSB, X-Axis Gyro MSB
data0 = bus.read_byte_data(0x6A, 0x28)
data1 = bus.read_byte_data(0x6A, 0x29)

# Convert the data
xGyro = data1 * 256 + data0
if xGyro > 32767 :
	xGyro -= 65536

# LSM330 Gyro address, 0x6A(106)
# Read data back from 0x2A(42), 2 bytes
# Y-Axis Gryo LSB, Y-Axis Gyro MSB
data0 = bus.read_byte_data(0x6A, 0x2A)
data1 = bus.read_byte_data(0x6A, 0x2B)

# Convert the data
yGyro = data1 * 256 + data0
if yGyro > 32767 :
	yGyro -= 65536

# LSM330 Gyro address, 0x6A(106)
# Read data back from 0x2C(44), 2 bytes
# Z-Axis Gyro LSB, Z-Axis Gyro MSB
data0 = bus.read_byte_data(0x6A, 0x2C)
data1 = bus.read_byte_data(0x6A, 0x2D)

# Convert the data
zGyro = data1 * 256 + data0
if zGyro > 32767 :
	zGyro -= 65536

# LSM330 Accl address, 0x1E(30)
# Select control register1, 0x20(32)
#		0x67(103)	Power ON, Data rate selection = 100 Hz
#					X, Y, Z-Axis enabled
bus.write_byte_data(0x1E, 0x20, 0x67)

time.sleep(0.5)

# LSM330 Accl address, 0x1E(30)
# Read data back from 0x28(40), 2 bytes
# X-Axis Accl LSB, X-Axis Accl MSB
data0 = bus.read_byte_data(0x1E, 0x28)
data1 = bus.read_byte_data(0x1E, 0x29)

# Convert the data
xAccl = data1 * 256 + data0
if xAccl > 32767 :
	xAccl -= 65536

# LSM330 Accl address, 0x1E()
# Read data back from 0x2A(42), 2 bytes
# Y-Axis Accl LSB, Y-Axis Accl MSB
data0 = bus.read_byte_data(0x1E, 0x2A)
data1 = bus.read_byte_data(0x1E, 0x2B)

# Convert the data
yAccl = data1 * 256 + data0
if yAccl > 32767 :
	yAccl -= 65536

# LSM330 Accl address, 0x1E(30)
# Read data back from 0x2C(44), 2 bytes
# Z-Axis Accl LSB, Z-Axis Accl MSB
data0 = bus.read_byte_data(0x1E, 0x2C)
data1 = bus.read_byte_data(0x1E, 0x2D)

# Convert the data
zAccl = data1 * 256 + data0
if zAccl > 32767 :
	zAccl -= 65536

# Output data to screen
#print ("X-Axis of Rotation : %d" % (xGyro))
#print ("Y-Axis of Rotation : %d" % (yGyro))
#print ("Z-Axis of Rotation : %d" % (zGyro))
print ("Acceleration in X-Axis : %f" % (xAccl*2*9.8/32767))
print ("Acceleration in Y-Axis : %f" % (yAccl*2*9.8/32767))
print ("Acceleration in Z-Axis : %f" % (zAccl*2*9.8/32767))
