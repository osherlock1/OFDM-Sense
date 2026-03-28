# How to collect data for experiments

This is a quick totorial for using the ```collect_raw_data.py``` script.

## Summary

This script allows you to easily collect data for experiments.  You can specify the name of directory for the experiment to be saved to. 

```
EXPERIMENTS/
└── {EXPERIMENT_NAME}/
    ├── fixed_positions.json
    └── {ROAMING DEVICE}archive/
        ├── position_{n}/
            ├── metadata.json
            ├── channel0/
            │   ├── {EXPERIMENT_NAME}_run_1_{timestamp}.dat
            │   ├── {EXPERIMENT_NAME}_run_2_{timestamp}.dat
            │   └── ...
            ├── channel1/
            │   ├── {EXPERIMENT_NAME}_run_1_{timestamp}.dat
            │   └── ...
            └── channel2/
```

## Pysical Set Up

Begin by position the devices in the desired orientation.  (For localization, you want to spread out devices to get better results).

### Verify data can be unpacked 
Once you are happy with the position of the Sivers, verify a packet can be recieved and unpacked by all channels.

1. Generate a data packet if not already done ```scripts/generate_packet.py -n 30```
2. Run transfer ```scripts/run_transfer.py```
3. Evaluate if data was succesfully recovered ```scripts/unpack_rx.py``` (add ```--plot``` for visualization)

If the evaluation metrics and constalation plots look good you are ready to move on to collecting data. YOU MUST VERIFY DATA CAN BE UNPACKED EVERYTIME YOU CHANGE THE PHYSICAL SET UP.


## Experiment Set Up

Inside the ```collect_raw_data.py``` script you will find these global variables

```python
# --------- MODIFY --------------
EXPERIMENT_NAME = "virtual_multilateration_3" # CHOOSE NAME OF EXPERIMENT TO BE RUN
ROAMING_DEVICES = ["RX2ch1"] # NAME OF DEVICE THAT IS MOVED (WILL ASK FOR POSITIONS EACH RUN)
FIXED_DEVICES = ["ANCHORch0", "TX"] # NAME OF DEVICES THAT ARE FIXED (WILL ONLY ASK ONCE PER EXPERIMENT)
# -------------------------------
```

These are the only three things you will need to modify for running experiments.

### **EXPERIMENT_NAME:**
 This will be the name of your experiment, it will save all of the data you collect and the positions of the devices into this directory. To make a new experiemnt change the name of this variable

### **FIXED_DEVICES**
 These are the name of the devices that will not move throughout the entire experiment. When you make a new experiment, the first time you run the ```collect_raw_data.py``` script it will make ```fixed_positions.json``` file storing the position of these devices.  As long as this file exists in the experiment dir it will not ask for these positions again.  

### **ROAMING_DEVICES:** 
These are the names of "virtual" RX devices that will be moved to simulate having more physical Sivers. **Changing the name of this global variable will save the data in a new run directory within the experiment directory**

**NOTE:** the FIXED_DEVICES and ROAMING_DEVICES can be named anything (as long as they follow propry file naming) they are not correlated to the acutaly USRP device or channels.  I prefer giving them names that will be easy to remember. i.e. if doing a run for the virtual RX3, name it RX3 and specify the channel that RX3 is recieving to. so RX3ch1.


# Example workflow
1. Change name of ```EXPERIMENT_NAME```
2. Set up physical set up
3. Verify all RX can recover sent data
4. Change name of fixed devices and roaming devices
5. run ```collect_raw_data.py```
6. take measurements and input data
7. Move roaming RX and change name of roaming devices
8. run ```collect_raw_data.py``` again 
