#------------------------------------------------------------------------------------------------------------------
#   Online classification of mobile sensor data
#------------------------------------------------------------------------------------------------------------------

import time
import requests
import numpy as np
import threading
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch
from joblib import load
import pandas as pd

##########################################
############ Data properties #############
##########################################

sampling_rate = 50      # Sampling rate in Hz of the input data
window_time = 1       # Window size in seconds for each trial window
window_samples = int(window_time * sampling_rate)   # Number of samples in each window

##########################################
##### Load data and train model here #####
##########################################

model = load('extra_trees_model_Alan.pkl')

##########################################
##### Data acquisition configuration #####
##########################################

# Communication parameters
IP_ADDRESS = '192.168.1.7'
COMMAND = 'accX&accY&accZ&acc_time'
BASE_URL = "http://{}/get?{}".format(IP_ADDRESS, COMMAND)

# Data buffer (circular buffer)
max_samp_rate = 5000            # Maximum possible sampling rate
n_signals = 3                   # Number of signals (accX, accY, accZ)
buffer_size = max_samp_rate*5   # Buffer size (number of samples to store)

buffer = np.zeros((buffer_size, n_signals + 1), dtype='float64')    # Buffer for storing data
buffer_index = 0                                                    # Index for the next data point to be written
last_sample_time = 0.0                                              # Last sample time for the buffer

# Flag for stopping the data acquisition
stop_recording_flag = threading.Event()

# Mutex for thread-safe access to the buffer
buffer_lock = threading.Lock()

# Function for continuously fetching data from the mobile device
def fetch_data():
    sleep_time = 1. / max_samp_rate 
    while not stop_recording_flag.is_set():
        try:
            response = requests.get(BASE_URL, timeout=0.5)
            response.raise_for_status()            
            data = response.json()

            global buffer, buffer_index, last_sample_time

            with buffer_lock:  # Ensure thread-safe access to the buffer
                buffer[buffer_index, 0] = data["buffer"]["acc_time"]["buffer"][0]    
                buffer[buffer_index, 1] = data["buffer"]["accX"]["buffer"][0]
                buffer[buffer_index, 2] = data["buffer"]["accY"]["buffer"][0]
                buffer[buffer_index, 3] = data["buffer"]["accZ"]["buffer"][0]

                buffer_index = (buffer_index + 1) % buffer_size
                last_sample_time = data["buffer"]["acc_time"]["buffer"][0] 

        except Exception as e:
            print(f"Error fetching data: {e}")

        time.sleep(sleep_time)

# Function for stopping the data acquisition
def stop_recording():
    stop_recording_flag.set()
    recording_thread.join()
    
# Start data acquisition
recording_thread = threading.Thread(target=fetch_data, daemon=True)
recording_thread.start()

##########################################
######### Online classification ##########
##########################################

def compute_frequency_features(signal):
    fft_vals = np.abs(rfft(signal))
    fft_freqs = rfftfreq(len(signal), 1.0 / sampling_rate)
    peak_freq = fft_freqs[np.argmax(fft_vals)]
    psd = np.abs(fft_vals)**2
    psd_norm = psd / np.sum(psd) if np.sum(psd) != 0 else psd
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    mean_freq = np.sum(fft_freqs * psd) / np.sum(psd) if np.sum(psd) != 0 else 0
    fft_energy = np.sum(psd)
    return peak_freq, spectral_entropy, mean_freq, fft_energy

update_time = 0.25
ref_time = time.time()

while True:
        
    time.sleep(update_time)   

    if buffer_index > 2*sampling_rate:
    
        ref_time = time.time()

        end_index = (buffer_index - 1) % buffer_size
        start_index = (buffer_index - 2) % buffer_size

        with buffer_lock:
            while (buffer[end_index, 0] - buffer[start_index, 0]) <= window_time:
                start_index = (start_index-1) % buffer_size

            indices = (buffer_index - np.arange(buffer_size, 0, -1)) % buffer_size            
            last_raw_data = buffer[indices, :]  # Get last data samples from the buffer

        t = last_raw_data[:, 0]  # Time vector from the buffer
        t_uniform = np.linspace(last_sample_time-window_time, last_sample_time, int(window_time * sampling_rate))   

        last_data = np.zeros((len(t_uniform), n_signals))  # Array with interpolated data
        for i in range(n_signals):
            interp_x = interp1d(t, last_raw_data[:, i+1], kind='linear', fill_value="extrapolate")
            last_data[:,i] = interp_x(t_uniform)

        #print ("Window data:\n", last_data)

        #######################################################
        ##### Calculate features of the last data samples #####
        #######################################################

        x, y, z = last_data[:, 0], last_data[:, 1], last_data[:, 2]
        features = [
            np.mean(x), np.std(x), kurtosis(x), skew(x),
            np.mean(y), np.std(y), kurtosis(y), skew(y),
            np.mean(z), np.std(z), kurtosis(z), skew(z),
            np.sqrt(np.mean(x**2 + y**2 + z**2)),
            np.median(x), np.min(x), np.max(x), np.ptp(x),
            np.median(y), np.min(y), np.max(y), np.ptp(y),
            np.median(z), np.min(z), np.max(z), np.ptp(z)
        ]

        for signal in [x, y, z]:
            features.extend(compute_frequency_features(signal))

        feature_names = [
            'mean_x', 'std_x', 'kurtosis_x', 'skew_x',
            'mean_y', 'std_y', 'kurtosis_y', 'skew_y',
            'mean_z', 'std_z', 'kurtosis_z', 'skew_z',
            'rms',
            'median_x', 'min_x', 'max_x', 'range_x',
            'median_y', 'min_y', 'max_y', 'range_y',
            'median_z', 'min_z', 'max_z', 'range_z',
            'peak_freq_x', 'spectral_entropy_x', 'mean_freq_x', 'fft_energy_x',
            'peak_freq_y', 'spectral_entropy_y', 'mean_freq_y', 'fft_energy_y',
            'peak_freq_z', 'spectral_entropy_z', 'mean_freq_z', 'fft_energy_z'
        ]

        features_df = pd.DataFrame([features], columns=feature_names)

        prediction = model.predict(features_df)

        if prediction[0] == 1:
            print("Activity: Resting")
        elif prediction[0] == 2:
            print("Activity: Jumping")
        elif prediction[0] == 3:
            print("Activity: Squatting")
        elif prediction[0] == 4:
            print("Activity: Side to side")
        elif prediction[0] == 5:
            print("Activity: Kicking")
        elif prediction[0] == 6:
            print("Activity: Walking")
        else:
            print("")

# Stop data acquisition
stop_recording()

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------
