import machine
import utime
import sys
import struct
import math

##Logistic Regression in Raspberry Pi Pico-RP-2040 using Micropython
##Use small file size to avoid memory error
##Tested using only 100 row in the diabetes.csv file, check file name, make sure
#that 1st row (column header of csv file) is removed (string not supported)


# scratch code for logistic regression in Micropython
# Numpy-like matrix library from scratch
# Created on 7/6/2022
# Note that, matrix must be two-dimensional
#Rev01: 7/6/22
#Rev02:11/6/22

def crossings_nonzero_all(data):
    prev = [0] #data > 0
    crossings = 0
    prev_idx = 0
    for i in data:
        if ((i > 0) and (prev[prev_idx] < 0)):
            crossings = crossings + 1
        elif ((i < 0) and (prev[prev_idx] > 0)):
            crossings = crossings + 1
            
        prev.append(i)
        prev_idx = prev_idx + 1
        
        
        
    return crossings


def diff_one_d(data):
    diff = []
    counter = 0
    idx = 0
    for i in data:
        if (counter == 0):
            counter = counter + 1
            continue
        
        diff.append(i - data[idx])
        idx = idx + 1
        
    return diff


def zeros(rows, cols):
    """
    Creates a matrix filled with zeros.
        :param rows: the number of rows the matrix should have
        :param cols: the number of columns the matrix should have
        :return: list of lists that form the matrix
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M


def zeros1d(x):  # 1d zero matrix
    z = [0 for i in range(len(x))]
    return z


def add1d(x, y):
    if len(x) != len(y):
        print("Dimention mismatch")
        exit()
    else:
        z = [x[i] + y[i] for i in range(len(x))]
        return z


def eye(n):
    """
    Creates and returns an identity matrix.
        :param n: the square size of the matrix
        :return: a square identity matrix
    """
    IdM = zeros(n, n)
    for i in range(n):
        IdM[i][i] = 1.0

    return IdM


def print_matrix(M, decimals=3):
    """
    Print a matrix one row at a time
        :param M: The matrix to be printed
    """
    for row in M:
        print([round(x, decimals) + 0 for x in row])


def transpose(M):
    """
    Returns a transpose of a matrix.
        :param M: The matrix to be transposed
        :return: The transpose of the given matrix
    """
    # Section 1: if a 1D array, convert to a 2D array = matrix
    if not isinstance(M[0], list):
        M = [M]

    # Section 2: Get dimensions
    rows = len(M)
    cols = len(M[0])

    # Section 3: MT is zeros matrix with transposed dimensions
    MT = zeros(cols, rows)

    # Section 4: Copy values from M to it's transpose MT
    for i in range(rows):
        for j in range(cols):
            MT[j][i] = M[i][j]

    return MT


def sub(x, y):  # 1d subtraction between two list
    if len(x) != len(y):
        print("Dimension mismatch")
        exit()
    else:
        z = [x[i] - y[i] for i in range(len(x))]
        return z


def dot(A, B):
    """
    Returns the product of the matrix A * B where A is m by n and B is n by 1 matrix
        :param A: The first matrix - ORDER MATTERS!
        :param B: The second matrix
        :return: The product of the two matrices
    """
    # Section 1: Ensure A & B dimensions are correct for multiplication
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = 1
    if colsA != rowsB:
        raise ArithmeticError('Number of A columns must equal number of B rows.')

    # Section 2: Store matrix multiplication in a new matrix
    C = zeros(rowsA, colsB)
    for i in range(rowsA):
        total = 0
        for ii in range(colsA):
            total += A[i][ii] * B[ii]
            C[i] = total

    return C


##Sigmoid function
def sigmoid(x):
    import math
    z = [1 / (1 + math.exp(-x[kk])) for kk in range(len(x))]
    return z


def binary_loss(ytrue, ypred):
    import math
    z = [-(float(ytrue[i]) * math.log(ypred[i])) - ((1 - float(ytrue[i])) * math.log(1 - ypred[i])) for i in
         range(len(ytrue))]
    cost = (1 / len(ytrue)) * sum(z)
    return cost


def evaluate_pred(w, x, b):
    print(x)
    # print(len(x[0]))
    tmp = zeros1d(x[0])
    for i in range(len(x)):
        tmp = add1d(tmp, [w[i] * x[i][j] for j in range(len(x[0]))])
    yp = sigmoid([tmp[i] + b for i in range(len(tmp))])
    return yp


##Logistic regression function
def logistic_regressor(x, y, lr, epoch):  ##lr:learning rate, niter:max iteration
    import random
    # global w, b
    w = []
    b = 0
    t = []

    for k in range(len(x)):
        ww = random.random()
        w.append(ww)

    # Gradient Descent algorithm
    for niter in range(epoch):  # looping upto no of epoch
        # Main logistic func part:f=W.TX+b
        # for j in range(len(x)):  # for no of feature
        # z = [w[j] * x[j][kk] for kk in range(len(x[0]))]  # wrong
        # z = add1d(z, [w[j] * x[j][kk] for kk in range(len(x[0]))])
        # Manual coding for 4 feature-testing
        # for i in range(len(x)):
        # w0 = [w[0] * x[0][kk] for kk in range(len(x[0]))]
        # w1 = [w[1] * x[1][kk] for kk in range(len(x[0]))]
        # w2 = [w[2] * x[2][kk] for kk in range(len(x[0]))]
        # w3 = [w[3] * x[3][kk] for kk in range(len(x[0]))]
        # z = add1d(w3, add1d(w2, add1d(w0, w1)))

        # add bias term 'b'
        # yp = sigmoid([z[i] + b for i in range(len(z))])

        # yp = sigmoid([z[i] + b for i in range(len(z))])
        # yp = sigmoid(z)  # predicted y
        yp = evaluate_pred(w, x, b)
        # print(yp[:5])
        # print(yp1[:5])
        # Derivative part
        dz = (1 / len(y)) * sum([yp[j] - y[j] for j in range(len(y))])
        # print(x)
        ff = dot(x, sub(yp, y))
        # print(ff)
        dw = [(1 / len(y)) * float(ff[j]) for j in range(len(ff))]
        db = dz

        for ii in range(len(x)):  # update weights
            w[ii] -= (lr * dw[ii])

        # update bias
        b -= (lr * db)
        # calculate loss
        loss = binary_loss(y, yp)
        print("No of epoch: " + str(niter))
        print("Training loss: " + str(loss))

    return w, b, loss, yp


# Prediction using trained model


def mean(x):  # calculate mean of an array or 1D matrix
    z = sum(x) / len(x)
    return z


def stdev(x):  # calculate std deviation of 1D array
    import math
    Xmean = sum(x) / len(x)
    N = len(x)
    tmp = 0
    for i in range(N):
        tmp = tmp + (x[i] - Xmean) ** 2
        z = math.sqrt(tmp / (N - 1))
    return z


def normalize(x):  # x is a 1d array
    nx = [(x[u] - mean(x)) / max(stdev(x),0.1) for u in range(len(x))]
    print("Normalize Data")
    print(mean(x))
    print(stdev(x))
    return nx


def predict_class(ypred):
    ypred_class = [1 if i > 0.5 else 0 for i in ypred]
    return ypred_class


def classification_report(ytrue, ypred):  # print prediction results in terms of metrics and confusion matrix
    tmp = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(ytrue)):
        if ytrue[i] == ypred[i]:  # For accuracy calculation
            tmp += 1
        ##True positive and negative count
        if ytrue[i] == 1 and ypred[i] == 1:  # find true positive
            TP += 1
        if ytrue[i] == 0 and ypred[i] == 0:  # find true negative
            TN += 1
        if ytrue[i] == 0 and ypred[i] == 1:  # find false positive
            FP += 1
        if ytrue[i] == 1 and ypred[i] == 0:  # find false negative
            FN += 1
    accuracy = tmp / len(ytrue)
    conf_matrix = [[TP, FP], [FN, TN]]
    #print(TP, FP, FN, TN)

    print("Accuracy: " + str(accuracy))
    print("Confusion Matrix:")
    print(print_matrix(conf_matrix))
    
# Function to split train and test set
def train_test_split(scaled_x_data,ydata, factor):
    scaled_data = transpose(scaled_x_data)
    N = len(scaled_data)
    print(N)
    n_sample = int(factor * N)
    print(n_sample)
    xtrain_set = transpose(scaled_data[:n_sample])
    xtest_set = transpose(scaled_data[n_sample:])
    ytrain_set = ydata[:n_sample]
    ytest_set = ydata[n_sample:]
    #print(len(xtrain_set))
    return xtrain_set, xtest_set, ytrain_set, ytest_set

##Logistic Regression in Raspberry Pi Pico-RP-2040 using Micropython
##Use small file size to avoid memory error
##Tested using only 100 row in the diabetes.csv file, check file name, make sure
#that 1st row (column header of csv file) is removed (string not supported)
#from pylab import logistic_regression as lm
def csvread(file_name):  # function for reading csv file
    f = open(file_name, 'r')
    w = []
    tmp = []
    for each in f:
        w.append(each)
        #print (each)

    # print(w)
    for i in range(len(w)):
        data = w[i].split(",")
        tmp.append(data)
        #print(data)
    file_data = transpose([[float(y) for y in x] for x in tmp])
    # file_data = [[float(y) for y in x] for x in tmp]
    return file_data
####### Test function ###########################
raw_data = csvread('data.csv')

# Normalize data using mean and stdev()

scaled_data = [normalize(raw_data[i]) for i in range(len(raw_data[:21]))]



def update_model(scaled_data):
    
    xtrain, xtest, ytrain, ytest = train_test_split(scaled_data,raw_data[21], 0.7)#70% data used for training, can change this
    # Xtrain data
    #Train and Build model using Logistic Regression
    w, b, loss_train, ypred_train = logistic_regressor(xtrain, ytrain, 0.1, 100)#1000 epoch, learning rate =0.1
    

    ypred_train = predict_class(ypred_train)
    classification_report(ytrain, ypred_train)
    #print(len(xtrain[0]))
    ##test set
    #W = [0.37799744,1.06052058,-0.25524181,0.04704248,-0.1428577,0.73400824,0.32438074,0.20997864] 
    #B = -0.82373324
    ypred_test = predict_class(evaluate_pred(w,xtest,b))
    classification_report(ytest, ypred_test)

    return w, b

w, b = update_model(scaled_data)




#Set up led
led_fall = machine.Pin(15, machine.Pin.OUT)
led_fall.low()

#Set up button
button = machine.Pin(14, machine.Pin.IN, machine.Pin.PULL_DOWN)


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
    
    print("scaled data info")
    print(len(scaled_data))
    print(len(scaled_data[0]))
    
    if slice_count == 12:
        slice_count = 0
        
        x_min = min(x_acc)
        y_min = min(y_acc)
        z_min = min(z_acc)
        
        x_arg_min = x_acc.index(x_min)
        y_arg_min = y_acc.index(y_min)
        z_arg_min = z_acc.index(z_min)
        
        x_max = max(x_acc)
        y_max = max(y_acc)
        z_max = max(z_acc)
        
        x_arg_max = x_acc.index(x_max)
        y_arg_max = y_acc.index(y_max)
        z_arg_max = z_acc.index(z_max)
        
        
        x_std = stdev(x_acc)
        y_std = stdev(y_acc)
        z_std = stdev(z_acc)
        
        
        x_mean = mean(x_acc)
        y_mean = mean(y_acc)
        z_mean = mean(z_acc)
        
        x_slope = mean(diff_one_d(x_acc))
        y_slope = mean(diff_one_d(y_acc))
        z_slope = mean(diff_one_d(z_acc))
        
        diff_one_d
        
        x_zc = crossings_nonzero_all(x_acc)
        y_zc = crossings_nonzero_all(y_acc)
        z_zc = crossings_nonzero_all(z_acc)
        
        
        
        x_mmd = math.sqrt((x_max-x_min)*(x_max-x_min) + (x_arg_max-x_arg_min)*(x_arg_max-x_arg_min))
        y_mmd = math.sqrt((y_max-y_min)*(y_max-y_min) + (y_arg_max-y_arg_min)*(y_arg_max-y_arg_min))
        z_mmd = math.sqrt((z_max-z_min)*(z_max-z_min) + (z_arg_max-z_arg_min)*(z_arg_max-z_arg_min))
        
        X_min_mean = -2.901961
        X_min_std = 6.385154

        Y_min_mean = -9.176471
        Y_min_std = 8.503425

        Z_min_mean = 0.254902
        Z_min_std = 7.010972

        X_max_mean = 5.72549
        X_max_std = 6.767801

        Y_max_mean = 5.254902
        Y_max_mean = 6.183342

        Z_max_mean = 9.921569
        Z_max_std = 7.008117

        X_std_mean = 1.137255
        X_std_std = 1.233201

        Y_std_mean = 2.470588
        Y_std_std = 2.508808

        Z_std_mean = 1.45098
        Z_std_std = 1.553238

        X_mean_mean = 1.392157
        X_mean_std =3.914478

        Y_mean_mean = -0.2156863
        Y_mean_std = 1.879508

        Z_mean_mean = 4.392157
        Z_mean_std = 4.582918


        X_zc_mean = 4.098039
        X_zc_std = 5.510917

        Y_zc_mean = 6.176471
        Y_zc_std = 8.721709

        Z_zc_mean = 2.901961
        Z_zc_std = 4.883666

        X_mmd_mean = 209.2941
        X_mmd_std = 226.3398

        Y_mmd_mean = 184.1961
        Y_mmd_std =140.8922

        Z_mmd_mean = 208.4902
        Z_mmd_std = 148.1164
                

        
        
        data = [[(x_min-X_min_mean)/X_min_std],[(y_min-Y_min_mean)/Y_min_std],[(z_min-Z_min_mean)/Z_min_std],[(x_max-X_max_mean)/X_max_std],[(y_max-Y_max_mean)/Y_max_mean],[(z_max-Z_max_mean)/Z_max_mean],[(x_std-X_std_mean)/X_std_std],[(y_std-Y_std_mean)/Y_std_std],[(z_std-Z_std_mean)/Z_std_mean],[(x_mean-X_mean_mean)/X_mean_std],[(y_mean-Y_mean_mean)/Y_mean_std],[(z_mean-Z_mean_mean)/Z_mean_std],[x_slope],[y_slope],[z_slope],[(x_zc-X_zc_mean)/X_zc_std],[(y_zc-Y_zc_mean)/Y_zc_std],[(z_zc-Z_zc_mean)/Z_zc_std],[(x_mmd-X_mmd_mean)/X_mmd_std],[(y_mmd-Y_mmd_mean)/Y_mmd_std],[(z_mmd-Z_mmd_mean)/Z_mmd_mean]]
        better_data = transpose(data)
        #scaled_data = [normalize(data[i]) for i in range(len(data[:21]))]
        print("This is the data we collected")
        print(data)
        print(better_data)
        
        print("Raw prediction")
        pred = evaluate_pred(w, data, b)
        print(pred)
        
        is_fall = predict_class(pred)
        print("Was this a fall?")
        
        if (int(is_fall[0]) == 1):
            print("FALL DETECTED")
            led_fall.high()
            
            
            for i in range(100):    
                if (button.value() == 1):
                    print("You Indicated False postive")
                    led_fall.low()
                    
                    #Fix label
                    #data[0][21] = 0
                    data.append([0])
                    
                    newdata = [(x_min-X_min_mean)/X_min_std,(y_min-Y_min_mean)/Y_min_std,(z_min-Z_min_mean)/Z_min_std,(x_max-X_max_mean)/X_max_std,(y_max-Y_max_mean)/Y_max_mean,(z_max-Z_max_mean)/Z_max_mean,(x_std-X_std_mean)/X_std_std,(y_std-Y_std_mean)/Y_std_std,(z_std-Z_std_mean)/Z_std_mean,(x_mean-X_mean_mean)/X_mean_std,(y_mean-Y_mean_mean)/Y_mean_std,(z_mean-Z_mean_mean)/Z_mean_std,x_slope,y_slope,z_slope,(x_zc-X_zc_mean)/X_zc_std,(y_zc-Y_zc_mean)/Y_zc_std,(z_zc-Z_zc_mean)/Z_zc_std,(x_mmd-X_mmd_mean)/X_mmd_std,(y_mmd-Y_mmd_mean)/Y_mmd_std,(z_mmd-Z_mmd_mean)/Z_mmd_mean]
                    #scaled_data.append(newdata)
                    
                    nidx=0
                    for j in range(21):
                        scaled_data[j].append(newdata[nidx])
                        nidx = nidx + 1
                        
                    raw_data[21].append(0)
                    break
                    
                utime.sleep(0.1)
                
                    
            #Add true positive
            #scaled_data.append([1])
            
            #retrain model
            print(newdata)
            print(scaled_data)
            print(raw_data[21])
            print("shapes")
            print(len(scaled_data))
            print(len(scaled_data[0]))
            w, b = update_model(scaled_data)
            
            
            
            
            
            
        else:
            print("No fall")
        
        
        
        x_acc = []
        y_acc = []
        z_acc = []
        
        
    slice_count = slice_count + 1
        
        
    
    
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
        
        
    pitch = 180 * math.atan(xAccl/math.sqrt(yAccl*yAccl + zAccl*zAccl))/(math.pi);
    roll = 180 * math.atan(yAccl/math.sqrt(xAccl*xAccl + zAccl*zAccl))/(math.pi);
    yaw = 180 * math.atan(zAccl/math.sqrt(xAccl*xAccl + zAccl*zAccl))/(math.pi);

    # Output data to screen
    #print("X-Axis of Rotation : %d" %xGyro)
    #print("Y-Axis of Rotation : %d" %yGyro)
    #print("Z-Axis of Rotation : %d" %zGyro)
    print("Acceleration in X-Axis : %d" %((xAccl*2*9.8)/32767))
    print("Acceleration in Y-Axis : %d" %((yAccl*2*9.8)/32767))
    print("Acceleration in Z-Axis : %d" %((zAccl*2*9.8)/32767))
    #print("Roll : %d" %roll)
    #print("Pitch : %d" %pitch)
    #print("Yaw : %d" %yaw)
    
    x_acc.append(((xAccl*2*9.8)/32767))
    y_acc.append(((yAccl*2*9.8)/32767))
    z_acc.append((-1*(zAccl*2*9.8)/32767))

    if (zAccl > 19344):
        print("(OLD) Looks like you fell!!!")

    
    led_onboard.value(1)
    utime.sleep(0.25)
    led_onboard.value(0)
    utime.sleep(0.25)