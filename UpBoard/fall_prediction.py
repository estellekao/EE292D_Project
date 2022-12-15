# Authors: Chris Calloway, Estelle Kao
# For Academic use only
# LSM330DLC code taken from vendor under free-will liscense 

import smbus
import time
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
import serial
import os
import math
import pickle
#import uselect


def init_variety_features():

    observed_labels_file = open("/home/ubuntu/Documents/EE292D_Project/feature_variation.pkl", "rb")
    dataset_features = pickle.load(observed_labels_file)

    return dataset_features


def compute_simlarity(dataset_features, current_queue ):

    total_similiarity = 0
    min_similiarity = 10000000

    for i in current_queue:
        #data_array_temp = [i['gyro_x_min'], i['gyro_y_min'], i['gyro_z_min'],i['gyro_x_max'],i['gyro_y_max'],i['gyro_z_max'],i['x_min'],i['y_min'],i['z_min'],i['x_max'],i['y_max'],i['z_max'],i['x_std'],i['y_std'],i['z_std'],i['x_mean'],i['y_mean'],i['z_mean'],i['x_slope'],i['y_slope'],i['z_slope'],i['x_zc'],i['y_zc'],i['z_zc'],i['x_mmd'],i['y_mmd'],i['z_mmd'], i['pitch_slope'], i['roll_slope']]
          
        A = np.array(i)
        # A = np.array()
        A_norm  = np.linalg.norm(A)
        min_similiarity = 10000000
        for j in dataset_features:

            data_array = [j['gyro_x_min'], j['gyro_y_min'], j['gyro_z_min'],j['gyro_x_max'],j['gyro_y_max'],j['gyro_z_max'],j['x_min'],j['y_min'],j['z_min'],j['x_max'],j['y_max'],j['z_max'],j['x_std'],j['y_std'],j['z_std'],j['x_mean'],j['y_mean'],j['z_mean'],j['x_slope'],j['y_slope'],j['z_slope'],j['x_zc'],j['y_zc'],j['z_zc'],j['x_mmd'],j['y_mmd'],j['z_mmd'], j['pitch_slope'], j['roll_slope']]
          
            B = np.array(data_array)
            A_dot_B = np.dot(A,B)
            B_norm  = np.linalg.norm(B)

            cosine_sim = A_dot_B / (A_norm * B_norm)
            
            
            if (cosine_sim < min_similiarity):
                min_similiarity = cosine_sim

        total_similiarity += min_similiarity



    return total_similiarity


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

    #Convert to mobi fall format (TODO uncomment)
    xAccl = (xAccl*9.80665)/(16666.6)
    yAccl = (yAccl*9.80665)/(16666.6)
    zAccl = (zAccl*9.80665)/(16666.6)

    return xAccl, yAccl, zAccl


def get_gyro_data():
    # Get I2C bus
    bus = smbus.SMBus(1)


    gyro_address = 106
    gyro_reg = 32

    #i2c_bus.write_byte_data(0x6A, 0x20, 0x0F)

    bus.write_byte_data(gyro_address,gyro_reg, 15)
    
     # LSM330 Gyro address, 0x6A(106)
    # Read data back from 0x28(40), 2 bytes
    # X-Axis Gyro LSB, X-Axis Gyro MSB
    data0 = bus.read_byte_data(gyro_address, 0x28)
    data1 = bus.read_byte_data(gyro_address, 0x29)

    # Convert the data
    xGyro = int((data1 * 256 + data0))
    if xGyro > 32767 :
        xGyro -= 65536

    # LSM330 Gyro address, 0x6A(106)
    # Read data back from 0x2A(42), 2 bytes
    # Y-Axis Gryo LSB, Y-Axis Gyro MSB
    data0 = bus.read_byte_data(gyro_address, 0x2A)
    data1 = bus.read_byte_data(gyro_address, 0x2B)

    # Convert the data
    yGyro = data1 * 256 + data0
    if yGyro > 32767 :
        yGyro -= 65536

    # LSM330 Gyro address, 0x6A(106)
    # Read data back from 0x2C(44), 2 bytes
    # Z-Axis Gyro LSB, Z-Axis Gyro MSB
    data0 = bus.read_byte_data(gyro_address, 0x2C)
    data1 = bus.read_byte_data(gyro_address, 0x2D)

    # Convert the data
    zGyro = data1 * 256 + data0
    if zGyro > 32767 :
        zGyro -= 65536


 


    #Correct for gyro sensitivy and convert to rad/s
    xGyro = (xGyro*math.pi)/(114.285*180)
    yGyro = (yGyro*math.pi)/(114.285*180)
    zGyro = (zGyro*math.pi)/(114.285*180)
    
    return xGyro, yGyro, zGyro





def get_mult_acc_data(time_interval = 6, data_spacing = 0.005, fake=False):
    x_acc = []
    y_acc = []
    z_acc = []
    x_gyro = []
    y_gyro = []
    z_gyro = []
    for i in range(0,int(time_interval/data_spacing)):
        # repeadly get accelerometer data
        xAccl, yAccl, zAccl = [0,0,0]
        xGyro, yGyro, zGyro = [0,0,0]
        if fake:
            xAccl = np.random.randint(20000) - 10000
            yAccl = np.random.randint(20000) - 10000
            zAccl = np.random.randint(20000) - 10000
            #print([xAccl,yAccl,zAccl])
        else:
            xAccl, yAccl, zAccl = get_acc_data()
            xGyro, yGyro, zGyro = get_gyro_data()

        x_acc.append(xAccl)
        y_acc.append(yAccl)
        z_acc.append(zAccl)
        x_gyro.append(xGyro)
        y_gyro.append(yGyro)
        z_gyro.append(zGyro)
        
        # sleep
        time.sleep(data_spacing)
    return x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro


# Calculates the number of times the sample crosses zero
# for one dimension
def zero_crossings(slice):
    return np.where(np.diff(np.signbit(slice)))[0].size

# Wafaa's metric
def min_max_distance(slice):
    return np.sqrt(np.square(np.amax(slice) - np.amin(slice))
                   + (np.square(np.argmax(slice) - np.argmin(slice))))


def calc_attitude_data(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
    # Initialise matrices and variables
    #C = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0] ])
    # P = np.eye(6)
    # Q = np.eye(6)
    # R = np.eye(2)

    #state_estimate = np.array([[0], [0], [0], [0], [0], [0]])

    C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    P = np.eye(4)
    Q = np.eye(4)
    R = np.eye(2)

    state_estimate = np.array([[0], [0], [0], [0]])

    phi_hat = 0.0
    theta_hat = 0.0

  



    # Calculate accelerometer offsets
    N = len(gyro_x)

    phi_hat_arr = [0]*N
    theta_hat_arr  = [0]*N
    yaw_hat_arr  = [0]*N
    phi_offset = 0.0
    theta_offset = 0.0

    for i in range(len(gyro_x)):
        phi_acc = math.atan2(acc_y[i], math.sqrt(acc_x[i] ** 2.0 + acc_z[i] ** 2.0))
        theta_acc = math.atan2(-acc_x[i], math.sqrt(acc_y[i] ** 2.0 + acc_z[i] ** 2.0))
        #[phi_acc, theta_acc] = gyro_x[i], gyro_y[i]
        phi_offset += phi_acc
        theta_offset += theta_acc
        #sleep(sleep_time)

    phi_offset = float(phi_offset) / float(N)
    theta_offset = float(theta_offset) / float(N)

    #print("Accelerometer offsets: " + str(phi_offset) + "," + str(theta_offset))


    # Measured sampling time
    dt = 0.0
    #start_time = time()

    #print("Running...")
    for i in range(len(gyro_x)):

        # Sampling time, 6e9 nanoseconds (or 6 seconds)
        dt = 6.0
        #start_time = time()

        # Get accelerometer measurements and remove offsets
        #[phi_acc, theta_acc] = imu.get_acc_angles()
        phi_acc = math.atan2(acc_y[i], math.sqrt(acc_x[i] ** 2.0 + acc_z[i] ** 2.0))
        theta_acc = math.atan2(-acc_x[i], math.sqrt(acc_y[i] ** 2.0 + acc_z[i] ** 2.0))
        phi_acc -= phi_offset
        theta_acc -= theta_offset
        
        # Gey gyro measurements and calculate Euler angle derivatives
        [p, q, r] = gyro_x[i], gyro_y[i], gyro_z[i]
        phi_dot = p + math.sin(phi_hat_arr[i]) * math.tan(theta_hat_arr[i]) * q + math.cos(phi_hat_arr[i]) * math.tan(theta_hat_arr[i]) * r
        theta_dot = math.cos(phi_hat_arr[i]) * q - math.sin(phi_hat_arr[i]) * r
        yaw_dot = math.sin(phi_hat_arr[i])*math.cos(theta_hat_arr[i])*q + math.sin(phi_hat_arr[i])*math.cos(theta_hat_arr[i])*r

        # Kalman filter
        # A = np.array([[1, 0, 0, -dt, 0, 0], [0, 1, 0, 0, -dt, 0], [0, 0, 1, 0 ,0, -dt], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1] ])
        # B = np.array([[dt, 0, 0], [0, dt, 0], [0, 0, dt], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        A = np.array([[1, -dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, -dt], [0, 0, 0, 1]])
        B = np.array([[dt, 0], [0, 0], [0, dt], [0, 0]])

        gyro_input = np.array([[phi_dot], [theta_dot]])
        state_estimate = A.dot(state_estimate) + B.dot(gyro_input)
        P = A.dot(P.dot(np.transpose(A))) + Q

        measurement = np.array([[phi_acc], [theta_acc]])
        y_tilde = measurement - C.dot(state_estimate)
        S = R + C.dot(P.dot(np.transpose(C)))
        K = P.dot(np.transpose(C).dot(np.linalg.inv(S)))
        state_estimate = state_estimate + K.dot(y_tilde)
        P = (np.eye(4) - K.dot(C)).dot(P)

        phi_hat_arr[i] = state_estimate[0]
        theta_hat_arr[i] = state_estimate[2]
        #yaw_hat_arr[i] = state_estimate[3]

        # Display results

        #print("Phi: " + str(round(phi_hat_arr[i][0] * 180.0 / math.pi, 1)) + " Theta: " + str(round(theta_hat_arr[i][0] * 180.0 / math.pi, 1)) + " YAW: " + str(round(yaw_dot * 180.0 / math.pi, 1)) )

    pitch_slope = np.mean(np.diff(phi_hat_arr))
    roll_slope = np.mean(np.diff(theta_hat_arr))

    if (math.isnan(pitch_slope)):
        pitch_slope = 0.1
    if (math.isnan(roll_slope)):
        roll_slope = 0.1
    #yaw_slope = np.mean(np.diff(acc_z))
    return pitch_slope, roll_slope

def preprocess_acc_data(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, pitch_slope, roll_slope):

    gyro_x_min = np.amin(gyro_x)
    gyro_y_min = np.amin(gyro_y)
    gyro_z_min = np.amin(gyro_z)

    gyro_x_max = np.amax(gyro_x)
    gyro_y_max = np.amax(gyro_y)
    gyro_z_max = np.amax(gyro_z)



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
    return [gyro_x_min, gyro_y_min, gyro_z_min, gyro_x_max, gyro_z_max, gyro_z_max, x_min, y_min, z_min, x_max, y_max, z_max, x_std, y_std, z_std, x_mean, y_mean, z_mean, x_slope, y_slope, z_slope, x_zc, y_zc, z_zc, x_mmd, y_mmd, z_mmd,  pitch_slope, roll_slope]

# Trains logistic regression on X_train and Y_train sets, predicts labels on X_test
# and Y_test sets, dumps model to file and returns predicted labels
def predict_fall(X_test, loaded_model, compare_version=False):

    
    Y_predict = loaded_model.predict(X_test)
    
    print(Y_predict)
    #print(loaded_model)
    return Y_predict


def train_fall_detection_model(X_train, Y_train, loaded_model):
    print("retraining model")
    print(Y_train)
    print("type of x-train:", type(X_train))
    print(X_train)
    print("type of y-train:", type(Y_train))
    print(Y_train)
    #Y_train = Y_train.reshape(-1,1)
    #model = MLPClassifier(random_state=1, max_iter=300, warm_start=False).partial_fit(X_train, Y_train, classes=np.array([0, 1]))
    #model = MLPClassifier(random_state=1, max_iter=300, warm_start=False).fit(X_train, Y_train)
    command = "TRAIN" + "\n"

    for i in range(100):
        ser.write(bytes(command.encode('ascii')))
        time.sleep(0.1)

    loaded_model.partial_fit(X_train, Y_train, loaded_model.classes_)

    # uncomment to save model
    filename = "../pkl_model/lr_pre_6.0E+09solver_lbfgsiter_1000run_retrain.pkl" 
    joblib.dump(loaded_model, filename)
    print("retrain completed")
    return 

if __name__ == "__main__":

    #INIT
    # TO-DO: existing model path is hard-coded at the moment. Make it a an argument.
    # TO-DO: No functionality of on-device training yet. Use Warm Start.
    filename = ""
    vanilla_filename = "../pkl_model/lr_pre_6.0E+09solver_lbfgsiter_1000run_vanilla.pkl"

    if False:
        filename = "../pkl_model/lr_pre_6.0E+09solver_lbfgsiter_1000run_10.pkl"
    else:
        filename = "../pkl_model/lr_pre_6.0E+09solver_lbfgsiter_1000run_retrain.pkl"
                #"pkl_model/lr_pre_" + prepro_param + "solver_" + str(solver) + "iter_" \
                #+ str(max_iter) + "run_" + str(run_count) + ".pkl"

    vanilla_loaded_model = joblib.load(vanilla_filename)
    vanilla_loaded_model.classes_ = [0,1]

    loaded_model = joblib.load(filename)
    loaded_model.classes_ = [0,1]

    dataset_features = init_variety_features()

    #END INIT

    new_X = []
    new_Y = []

    new_X_FP = []
    new_X_NP = []
    new_Y_FP = []
    new_Y_NP = []


    running_fps = 0

    serial_connected = 0
    ser = ""
    # Requires Pico to be plugged in and on
    if os.path.exists('/dev/ttyACM0') == True:
        ser = serial.Serial('/dev/ttyACM0', 115200)
        serial_connected = 1
        time.sleep(1)

    while True:
        # repeatedly get accelerometer data. wait time.

        # Display an alive signal to PICO
        command = "ALIVE" + "\n"
        ser.write(bytes(command.encode('ascii')))
        
        
        x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro = get_mult_acc_data(time_interval = 3, data_spacing = 0.001, fake=False)
        #print(x_acc, y_acc, z_acc)

        # preprocess the accelerometer data, and store into list format.
        # run_one_LR expects 2D-array. Adding another outer [] to make the 1D-array two dimensional.
        pitch_slope, roll_slope = calc_attitude_data(x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro)
        print(pitch_slope)
        print(roll_slope)
       
        X_test = [preprocess_acc_data(x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro,  pitch_slope, roll_slope)]
        
        #print(X_test)
        #print("len of X_train list: %s" % len(X_train))

        # get prediction
        Y_vanilla_predict = predict_fall(X_test, vanilla_loaded_model)[0]

        
        
        Y_predict = predict_fall(X_test, loaded_model)[0]


        overrule = 0
        if (ser.inWaiting() > 0):
                    # read the bytes and convert from binary array to ASCII
                    pico_data = ser.read(ser.inWaiting()).decode('ascii')
                    #pico_data = read_serial_input() #ser.readline()
                    #pico_data = pico_data.decode("utf-8","ignore")
                    print(pico_data[:-2])
                
                    if ("FN" in pico_data):
                        print("Self labeled fall!")
                        Y_predict = 1
                        overrule = 1


        print("Current Prediction is: " + str(Y_predict))
        print("Vanilla Prediction is: " + str(Y_vanilla_predict))

        if Y_vanilla_predict:
            command = "VANILLA" + "\n"
            ser.write(bytes(command.encode('ascii')))
            time.sleep(3)


        
        new_X.append(X_test[0])


        # update model
        # We get labeled data from the user
        # use Y_train=1 for fall, Y_train=0 for no fall. 
        val = 0

        FP = False

        if Y_predict:
            #val = input("Was that a fall? (1=Yes, 0=No)")
            # Assume True positive
            val = 1
            
            # New: write to Pico
            command = "LED_ON" + "\n"
            ser.write(bytes(command.encode('ascii')))
            

            # Allow user 10 seconds to label data
            stop_early = False
            for i in range(10):

                if (ser.inWaiting() > 0 and not stop_early):
                    # read the bytes and convert from binary array to ASCII
                    pico_data = ser.read(ser.inWaiting()).decode('ascii')
                    #pico_data = read_serial_input() #ser.readline()
                    #pico_data = pico_data.decode("utf-8","ignore")
                    print(pico_data[:-2])
                
                    if ("FP" in pico_data):
                        stop_early = True
                        FP = True
                        val = 0
                        
                        break
                time.sleep(1)
            
            print("This is val: " + str(val))
            print("typd of val:", type(val))
            new_Y.append(val)

            # train model and clear cache.
            # TO-DO: this is not good.... this is retrainig the entire model with only limited dataset from new_X and new_Y.
            #uncommenting out for now.
            #print(new_X)
    

        else:
            new_Y.append(val)

        
        if FP or overrule:
            running_fps = running_fps + 1
            new_X_FP.append(X_test[0])
            new_Y_FP.append(val)
            print(new_Y_FP)

        else:
            if (len(new_X_NP) == 5):
                new_X_NP.pop(0)
                new_X_NP.append(X_test[0])
                new_Y_NP.pop(0)
                new_Y_NP.append(val)
                
                print("THIS IS IS SIMLARITY")
                print(sim)
                print("!!!!!!!!!!!!!!!!!!!!!")

                print(new_Y_NP)

            else:
                new_X_NP.append(X_test[0])
                new_Y_NP.append(val)
                print(new_Y_NP)


        c_X = new_X_NP + new_X_FP
        sim = compute_simlarity(dataset_features, c_X )

        sim_hyper_param = 5.5
        #TODO Add hueristic to balance data types
        if (running_fps == 5  ):
            print("Retrain activated")
            c_X = new_X_NP + new_X_FP
            c_Y = new_Y_NP + new_Y_FP
            train_fall_detection_model(X_train=c_X, Y_train=c_Y, loaded_model=loaded_model)
            new_X = []
            new_Y = []
            new_X_FP = []
            new_X_NP = []
            new_Y_FP = []
            new_Y_NP = []
            running_fps = 0





