# Distributed with a free-will license.
# Use it any way you want, profit or free, provided it fits in the licenses of its associated works.
# LSM330
# This code is designed to work with the LSM330_I2CS I2C Mini Module available from ControlEverything.com.
# https://www.controleverything.com/content/Accelorometer?sku=LSM330_I2CS#tabs-0-product_tabset-2

import smbus
import time
import numpy as np
import joblib

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

    # TO-DO: returning the integer representation here. 
    # TO-DO: retrain .pkl model with int. Step1: scale the input by *32767/(2*9.8) from the MobileFall Dataset.
    return xAccl, yAccl, zAccl

def get_mult_acc_data(wait_time = 0.5):
    x_acc = []
    y_acc = []
    z_acc = []
    for i in range(0,12):
        # repeadly get accelerometer data
        xAccl, yAccl, zAccl = get_acc_data()
        x_acc.append(xAccl)
        y_acc.append(yAccl)
        z_acc.append(zAccl)
        
        # sleep
        time.sleep(wait_time)
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

    return [x_min, y_min, z_min, x_max, y_max, z_max, x_std, y_std, z_std, x_mean, y_mean, z_mean, x_slope, y_slope, z_slope, x_zc, y_zc, z_zc, x_mmd, y_mmd, z_mmd]

# Trains logistic regression on X_train and Y_train sets, predicts labels on X_test
# and Y_test sets, dumps model to file and returns predicted labels
def run_one_LR(X_test, rand_seed=None
               , solver='lbfgs', max_iter=1000, multi_class='ovr'
               , verbose=1, n_jobs=4, run_count=1, load_existing_model=1):

    # TO-DO: existing model path is hard-coded at the moment. Make it a an argument.
    # TO-DO: No functionality of on-device training yet. Use Warm Start.
    if load_existing_model:
        filename = "../pkl_model/lr_pre_6.0E+09solver_lbfgsiter_1000run_1.pkl"
                   #"pkl_model/lr_pre_" + prepro_param + "solver_" + str(solver) + "iter_" \
                   #+ str(max_iter) + "run_" + str(run_count) + ".pkl"
        loaded_model = joblib.load(filename)
        Y_predict = loaded_model.predict(X_test)
        print(Y_predict)
        print(loaded_model)
        return Y_predict

if __name__ == "__main__":

    # repeatedly get accelerometer data. wait time.
    x_acc, y_acc, z_acc = get_mult_acc_data(wait_time = 0.5)
    #print(x_acc, y_acc, z_acc)

    # preprocess the accelerometer data, and store into list format.
    # run_one_LR expects 2D-array. Adding another outer [] to make the 1D-array two dimensional.
    X_test = [preprocess_acc_data(x_acc, y_acc, z_acc)]
    #print(X_train)
    #print("len of X_train list: %s" % len(X_train))

    # get prediction
    run_one_LR(X_test, rand_seed=None
               , solver='lbfgs', max_iter=1000, multi_class='ovr'
               , verbose=1, n_jobs=4, run_count=1, load_existing_model=1)
