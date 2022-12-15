## Abstract

Falls represent a considerable medical danger to the world's population. Many fall detection systems have been implemented to rectify this issue, such as the Apple Watch fall detection, Life Alert, and various smartphone IMU applications. However, the models implemented in these devices are either inference-only or cloud-based. Given the diversity of lifestyles and living conditions in the world, a model that can retrain from data it collects on the edge could provide users with a more robust fall prediction system. This report presents an experimental device setup and data collection to test such an adaptive fall detection model on the edge. The device is initially trained on data from the MobiAct dataset and then retrained on the edge using data collected from an LSM330DLC Accelerometer/Gyroscope [1]. Our neural network-based device retrained on the edge achieved an accuracy of 71.0\% and 80.0\% on two activities we taught our model to label as a non-fall with only 1 and 3 retraining events respectively. We also achieved an accuracy 94.0\% on one new activity we taught the model to consider a fall with 4 retraining events.


## Scripts in this directory folder
### Purpose: train the 1-layer or 3-layer MLP model off-line.
The MLP model is adapted from [ttshiz's git repo](https://github.com/ttshiz/CS539_Fall_Prediction).

| Steps       | Script        | Description | Input     | Output
| :---        |    :----     |    :----  |          :---- |          :---- |
| Step#1      | preprocess.py | <ul>  <li>Process <MobiAct_Dataset_v2.0> Annotated data into .json fomat. </li>  <li>modifications: The accelerator and gyroscpe data is originally recorded in floating point and is casted to int after running this script</li> </ul>       |  <ul>  <li>ss: slice_size in nanoseconds. Determines the time stamp spacing to get new data </li>  <li>mobiact_folder: where your MobiAct dataset is stored</li> </ul>  | <ul><li>  preprocessed_*.json: preprocessed data (int casted)</li></ul>  |
| Step#2      | nn.py    |  <ul><li>Trains the model in K=10 fold. Binary Classification. </li><li>modification: Script was modified to allow new model training vs. loading a pre-trained model </li> </ul> | <ul>  <li>ss: filese: which preprocessed.json file to train and predict </li>  <li>load_existing_model: <ul><li>set to 1 if you had previously trained a model and what to use that, e.g. lr_pre_6.0E+09solver_lbfgsiter_1000run_1.pkl </li><li>set to 0 if you wish to retrain the model or that no model exist yet </li></ul></li> </ul> |<ul><li> lr_pre_*.pkl: model </li><li>bin_results_summary.json: model accuracy, F1score, precision, .... </li></ul> |

## Scripts in this Micropython folder
### Purpose: to set up push button and LED indicator on the Raspberry Pi Pico to communicate with the UpBoard.
Addition set up instructions are descrbied in [Micropython/README.md](https://github.com/estellekao/EE292D_Project/blob/main/Micropython/README.md)

## Scripts in this UpBoard folder
### Purpose: to set up on-device inference and training on the UpBoard. Instructions also include how to communicate with the LSM330DLC Accelerometer/Gyroscope module via I2C ports.
Addition set up instructions are descrbied in [UpBoard/README.md](https://github.com/estellekao/EE292D_Project/blob/main/UpBoard/README.md)

## Reference on MobiAct dataset
[1] Chatzaki C., Pediaditis M., Vavoulas G., Tsiknakis M. (2017) Human Daily Activity and Fall Recognition Using a Smartphone’s Acceleration Sensor. In: Rocker C., O’Donoghue J., Ziefle M., Helfert M., Molloy W. (eds) Information and Communication Technologies for Ageing Well and e-Health. ICT4AWE 2016. Communications in Computer and Information Science, vol 736, pp 100-118. Springer, Cham, DOI 10.1007/978-3-319-62704-5_7.

[2] Shizume, T., &amp; Salem, W. (n.d.). TTSHIZ/CS539_FALL_PREDICTION. GitHub. Retrieved December 14, 2022, from https://github.com/ttshiz/CS539_Fall_Prediction 
