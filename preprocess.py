import csv
import glob
import json
import numpy as np
import sklearn as sk

from collections import Counter

FALL_LABELS = set(['FOL', 'FKL', 'BSC', 'SDL'])

# Calculates the number of times the sample crosses zero
# for one dimension
def zero_crossings(slice):
    return np.where(np.diff(np.signbit(slice)))[0].size

# Wafaa's metric
def min_max_distance(slice):
    return np.sqrt(np.square(np.amax(slice) - np.amin(slice))
                   + (np.square(np.argmax(slice) - np.argmin(slice))))
    
# Calculates metrics for each dimension
def process_slice(slice):
    #print(slice)
    def slice_get(name):
        return [float(v[name]) for v in slice]
    slice_features = dict()

    slice_features["gyro_x_min"] = np.amin(slice_get("gyro_x"))
    slice_features["gyro_y_min"] = np.amin(slice_get("gyro_y"))
    slice_features["gyro_z_min"] = np.amin(slice_get("gyro_z"))
    slice_features["gyro_x_max"] = np.amax(slice_get("gyro_x"))
    slice_features["gyro_y_max"] = np.amax(slice_get("gyro_y"))
    slice_features["gyro_z_max"] = np.amax(slice_get("gyro_z"))

    # slice_features["roll_mean"] = np.amin(slice_get("roll"))
    # slice_features["pitch_mean"] = np.amin(slice_get("pitch"))
    # slice_features["azimuth_mean"] = np.amin(slice_get("azimuth"))
   




    slice_features["x_min"] = np.amin(slice_get("acc_x"))
    slice_features["y_min"] = np.amin(slice_get("acc_y"))
    slice_features["z_min"] = np.amin(slice_get("acc_z"))
    
    slice_features["x_max"] = np.amax(slice_get("acc_x"))
    slice_features["y_max"] = np.amax(slice_get("acc_y"))
    slice_features["z_max"] = np.amax(slice_get("acc_z"))
    
    slice_features["x_std"] = np.std(slice_get("acc_x"))
    slice_features["y_std"] = np.std(slice_get("acc_y"))
    slice_features["z_std"] = np.std(slice_get("acc_z"))
    
    slice_features["x_mean"] = np.mean(slice_get("acc_x"))
    slice_features["y_mean"] = np.mean(slice_get("acc_y"))
    slice_features["z_mean"] = np.mean(slice_get("acc_z"))

    slice_features["x_slope"] = np.mean(np.diff(slice_get("acc_x")))
    slice_features["y_slope"] = np.mean(np.diff(slice_get("acc_y")))
    slice_features["z_slope"] = np.mean(np.diff(slice_get("acc_z")))

    slice_features["x_zc"] = zero_crossings(slice_get("acc_x"))
    slice_features["y_zc"] = zero_crossings(slice_get("acc_y"))
    slice_features["z_zc"] = zero_crossings(slice_get("acc_z"))

    slice_features["x_mmd"] = min_max_distance(slice_get("acc_x"))
    slice_features["y_mmd"] = min_max_distance(slice_get("acc_y"))
    slice_features["z_mmd"] = min_max_distance(slice_get("acc_z"))


    slice_features["pitch_slope"] = np.mean(np.diff(slice_get("pitch")))
    slice_features["roll_slope"] = np.mean(np.diff(slice_get("roll")))
    #slice_features["z_slope"] = np.mean(np.diff(slice_get("acc_z")))

    # label each timeslice with the label of the majority class unless it is a fall
    #falls = FALL_LABELS.intersection(set([v["label"] for v in slice]))
    #slice_features["label"] = falls.pop() if falls else Counter(
    #    [v["label"] for v in slice]).most_common(1)[0][0]

    # label each timeslice as either fall or no fall.
    falls = FALL_LABELS & (set([v["label"] for v in slice]))
    slice_features["label"] = 1 if len(falls)!=0 else 0
    #if len(falls)!=0: print(slice_features)

    return slice_features

# Preprocesses one file for a given slice_size (nanoseconds)
def process_file(file_name, slice_size):
    try:
        #data = np.genfromtxt(file_name, dtype=None, delimiter=',', names=True)
        with open(file_name, "r") as csv_file:
            data = csv.DictReader(csv_file)
            feature_list = []
            snum = 1
            cur_slice = list()
            start_time = None
            for d in data:
                if not start_time:
                    start_time = float(d["timestamp"])
                # if we still need to find the index of the end of the slice
                if (float(d["timestamp"]) < start_time + snum*slice_size):
                    cur_slice.insert(0, d)
                # otherwise consider slice complete and process
                else:
                    cur_slice.reverse()
                    feature_list.insert(0, process_slice(cur_slice))
                    cur_slice = list([d])
                    snum = snum + 1
            else:
                if cur_slice:
                    cur_slice.reverse()
                    feature_list.insert(0, process_slice(cur_slice))
                    cur_slice = list()
                    snum = snum + 1
            feature_list.reverse()
            return feature_list
    except Exception as e:
        raise e
        print("Error Processing "+file_name + " " + str(e))
        return False

# Preprocesses the dataset directory for a given slice_size (nanoseconds)
def process_directory(slice_size, mobiact_folder):
    data = dict(slice_size=slice_size)
    print(data)
    for file in glob.glob(mobiact_folder + "*/*_annotated.csv", recursive=True):
        print(file)
        data[file] = process_file(file, slice_size)
    return data

def write_to_json(data):
    print("here?")
    with open("preprocessed_"+("%.1E"%data["slice_size"])+".json", "w") as fp:
        json.dump(data, fp)


# main function to enable running in terminal
def main(slice_size, mobiact_folder):
    
    print("Processing Directory")
    data = process_directory(slice_size, mobiact_folder)
    print(data)
    print("Writing File")
    write_to_json(data)

if __name__ == "__main__":
    ss =  6e9 #5.0e9 #2.5e9 #1e9 #sys.argv[1]; # Preprocesses the dataset directory for a given slice_size (nanoseconds)
    mobiact_folder = "/home/ubuntu/Documents/MobiAct_Dataset_v2.0/" 
    main(ss, mobiact_folder)
