# Distributed with a free-will license.
# Use it any way you want, profit or free, provided it fits in the licenses of its associated works.
# LSM330
# This code is designed to work with the LSM330_I2CS I2C Mini Module available from ControlEverything.com.
# https://www.controleverything.com/content/Accelorometer?sku=LSM330_I2CS#tabs-0-product_tabset-2

import smbus
import time
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
import serial

serial_connected = 0
ser = ""
# Requires Pico to be plugged in and on
if os.path.exists('/dev/ttyACM0') == True:
    ser = serial.Serial('/dev/ttyACM0', 115200)
    serial_connected = 1
    time.sleep(1)


def get_acc_data():
    # Get I2C bus
    bus = smbus.SMBus(1)

    # LSM330 Accl address, 0x1E(30)
    # Select control register1, 0x20(32)
    #		0x67(103)	Power ON, Data rate selection = 100 Hz
    #					X, Y, Z-Axis enabled
    bus.write_byte_data(0x1E, 0x20, 0x67)


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
    #print ("Acceleration in X-Axis : %f" % (xAccl*2*9.8/32767))
    #print ("Acceleration in Y-Axis : %f" % (yAccl*2*9.8/32767))
    #print ("Acceleration in Z-Axis : %f" % (zAccl*2*9.8/32767))

    # DONE: returning the integer representation here. 
    # DONE: retrain .pkl model with int. Step1: scale the input by *32767/(2*9.8) from the MobileFall Dataset.

    #This is for human reading [m/s^2]
    #xAccl = xAccl *2*9.8/32767
    #yAccl = yAccl *2*9.8/32767
    #zAccl = zAccl *2*9.8/32767
    return xAccl, yAccl, zAccl

def get_mult_acc_data(time_interval = 6, data_spacing = 0.005, fake=False):
    x_acc = []
    y_acc = []
    z_acc = []
    for i in range(0,int(time_interval/data_spacing)):
        # repeadly get accelerometer data
        if fake:
            xAccl = np.random.randint(20000) - 10000
            yAccl = np.random.randint(20000) - 10000
            zAccl = np.random.randint(20000) - 10000
            #print([xAccl,yAccl,zAccl])
        else:
            xAccl, yAccl, zAccl = get_acc_data()
        x_acc.append(xAccl)
        y_acc.append(yAccl)
        z_acc.append(zAccl)
        
        # sleep
        time.sleep(data_spacing)
    return x_acc, y_acc, z_acc


# Calculates the number of times the sample crosses zero
# for one dimension
def zero_crossings(slice):
    return np.where(np.diff(np.signbit(slice)))[0].size

# Wafaa's metric
def min_max_distance(slice):
    return np.sqrt(np.square(np.amax(slice) - np.amin(slice))
                   + (np.square(np.argmax(slice) - np.argmin(slice))))

def preprocess_acc_data(acc_x, acc_y, acc_z):
    x_min = np.amin(acc_x)
    y_min = np.amin(acc_y)
    z_min = np.amin(acc_z)

    x_max = np.amax(acc_x)
    y_max = np.amax(acc_y)
    z_max = np.amax(acc_z)

    x_std = np.std(acc_x)
    y_std = np.std(acc_y)
    z_std = np.std(acc_z)

    x_mean = np.mean(acc_x)
    y_mean = np.mean(acc_y)
    z_mean = np.mean(acc_z)

    x_slope = np.mean(np.diff(acc_x))
    y_slope = np.mean(np.diff(acc_y))
    z_slope = np.mean(np.diff(acc_z))

    x_zc = zero_crossings(acc_x)
    y_zc = zero_crossings(acc_y)
    z_zc = zero_crossings(acc_z)

    x_mmd = min_max_distance(acc_x)
    y_mmd = min_max_distance(acc_y)
    z_mmd = min_max_distance(acc_z)

    print("z_slope: ", z_slope)
    return [x_min, y_min, z_min, x_max, y_max, z_max, x_std, y_std, z_std, x_mean, y_mean, z_mean, x_slope, y_slope, z_slope, x_zc, y_zc, z_zc, x_mmd, y_mmd, z_mmd]

# Trains logistic regression on X_train and Y_train sets, predicts labels on X_test
# and Y_test sets, dumps model to file and returns predicted labels
def predict_fall(X_test, compare_version=False):

    # TO-DO: existing model path is hard-coded at the moment. Make it a an argument.
    # TO-DO: No functionality of on-device training yet. Use Warm Start.
    filename = ""

    if compare_version:
        filename = "../pkl_model/lr_pre_6.0E+09solver_lbfgsiter_1000run_10.pkl"
    else:
        filename = "../pkl_model/lr_pre_6.0E+09solver_lbfgsiter_1000run_retrain.pkl"
                #"pkl_model/lr_pre_" + prepro_param + "solver_" + str(solver) + "iter_" \
                #+ str(max_iter) + "run_" + str(run_count) + ".pkl"
    loaded_model = joblib.load(filename)
    Y_predict = loaded_model.predict(X_test)
    print(Y_predict)
    #print(loaded_model)
    return Y_predict


def train_fall_detection_model(X_train, Y_train):
    model = MLPClassifier(random_state=1, max_iter=300, warm_start=True).partial_fit(X_train, Y_train, ['1'])

    # uncomment to save model
    filename = "../pkl_model/lr_pre_6.0E+09solver_lbfgsiter_1000run_retrain.pkl" 
    joblib.dump(model, filename)
    return 

if __name__ == "__main__":

    new_X = []
    new_Y = []
    while True:
        # repeatedly get accelerometer data. wait time.
        x_acc, y_acc, z_acc = get_mult_acc_data(time_interval = 3, data_spacing = 0.001, fake=False)
        #print(x_acc, y_acc, z_acc)

        # preprocess the accelerometer data, and store into list format.
        # run_one_LR expects 2D-array. Adding another outer [] to make the 1D-array two dimensional.
        X_test = [preprocess_acc_data(x_acc, y_acc, z_acc)]
        #print(X_test)
        #print("len of X_train list: %s" % len(X_train))

        # get prediction
        
        Y_predict = predict_fall(X_test)
        print("Current Prediction is: " + str(Y_predict))
        new_X.append(X_test)

        # update model
        # We get labeled data from the user
        # use Y_train=1 for fall, Y_train=0 for no fall. 
        if Y_predict:
            #val = input("Was that a fall? (1=Yes, 0=No)")
            # Assume True positive
            val = 1
            
            # New: write to Pico
            command = "LED_ON" + "\n"
            ser.write(bytes(command.encode('ascii')))
            
            # Allow user 10 seconds to label data
            for i in range(10):
                pico_data = ser.readline()
                pico_data = pico_data.decode("utf-8","ignore")
                print(pico_data[:-2])
                if (pico_data == "FP"):
                    val = 0

                time.sleep(1)
                
            command = "LED_OFF" + "\n"
            ser.write(bytes(command.encode('ascii')))
                
            
            
            print(val)
            new_Y.append(Y_predict)

            # train model and clear cache.
            # TO-DO: this is not good.... this is retrainig the entire model with only limited dataset from new_X and new_Y.
            #uncommenting out for now.
            print(new_X)
            print(len(new_X))

            #TODO Add hueristic to balance data types
            if (len(new_X) == 5):
                train_fall_detection_model(X_train=new_X[0], Y_train=new_Y[0])
                new_X = []
                new_Y = []
        else:
            new_Y.append(Y_predict)


