# =============================================================================
#
# Machine Learning for Acoustic Anomaly Detection
# Feature Extraction (FE) program (version 2.4)
# Alex Renaudin 31 October 2023 
# 
# Computes the MFCCs of the audio in the 'AAD/audio' folder. 
# Program returns CSV file containing Mel band information + class indicator.
# Note some audio files (marked with '*') are filtered out due to mislabbeling.
#
# Audio filenames (e.g. '158_A.WAV') contain a number and a class:
# S = 'satisfactory' (i.e. normal) behaviour, A = 'anomalous' behaviour
#
# Tuneable program parameters:
#    - Sample rate
#    - Number of Mel bands 
#    - Frame size
#    - Hop size
##
# Uses the mean as statistical integration method. Alternate methods can be 
# introduced (e.g. max, RMS) for varied results. 
# 
# Based off 'Audio Signal Processing for Machine Learning' by Valerio Velardo
# https://www.youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0
#
# =============================================================================

import os
import librosa
import numpy as np
import csv
import glob
import random

# Program parameters:
SAMPLE_RATE = 8000      # Sample rate (kHz)
N_MELS = 64             # Number of Mel bands
FRAME_SIZE = 256        # Number of samples in frame
HOP_SIZE = 128          # Number of samples in hop

folder_path = os.path.join(os.getcwd(), "audio") # AAD audio folder
file_extension = ".wav" # Define file extension as WAV

# List audio files in the 'AAD/audio' folder using glob module
audio_files = sorted(glob.glob(folder_path + "/*" + file_extension))

# Filter out uncertain audio files marked with '*'
audio_files = [file for file in audio_files if not 
               file.endswith('*' + file_extension)]

random.shuffle(audio_files) # Shuffle list of audio files
num_files = len(audio_files) # Number of files in AAD folder

# Initalise feature (output) matrices (+1 column for class indicator)
dataset = np.zeros((num_files, N_MELS+1))

# Prints program information
print("\nRunning feature extraction program with the following parameters...")
print(f"\n\tTotal files:\t\t{num_files}")
print(f"\tSamplerate:\t\t{SAMPLE_RATE} Hz")
print(f"\tFrame size:\t\t{FRAME_SIZE}")
print(f"\tHop size:\t\t{HOP_SIZE}")
print(f"\tMel bands:\t\t{N_MELS}")

#%% Iterate through each audio file in the 'AAD/audio' folder
for n in range(num_files): 
    
    # Extract the class from the filename and add to last column of output
    class_label_str = audio_files[n].split("/")[-1].split(".")[0][-1]
    if class_label_str == 'A':
        dataset[n, N_MELS] = int(1)
    else:
        dataset[n, N_MELS] = int(0)
   
    # Librosa audio extraction
    audio, sr = librosa.load(audio_files[n], sr=SAMPLE_RATE) 
    
    #%% MFCC computation
    
    # Short-time fourier transform (STFT) from librosa library
    S_scale = librosa.stft(audio,n_fft=FRAME_SIZE,hop_length=HOP_SIZE)
    
    Y_scale = np.abs(S_scale) ** 2 # Square absolute S_scale value        
    Y_log_scale = librosa.power_to_db(Y_scale) # Convert to dB scale

    # Creates Mel filter banks
    filter_banks = librosa.filters.mel(n_fft=FRAME_SIZE, 
                                                sr=SAMPLE_RATE, n_mels=N_MELS)
    
    # Builds Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, 
                  sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=N_MELS)
    
    # Convert to dB scale
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
            
    # Calls librosa function to extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, n_mfcc=N_MELS, sr=sr)
    
    #%% Mean statistical indicator

    # Iterates through Mel bands and integrates using statistical indicators
    for i in range(N_MELS): 
        
        # Temporary values for mean and RMS respectively
        temp_sum = 0
        
        # Iterates through time of each Mel band and computes sums
        for j in range(mfccs.shape[1]): 
            temp_sum += mfccs[i, j]
        
        dataset[n, i] = temp_sum/mfccs.shape[1] # Final mean value
       
#%% Export output as .csv files

header = [f'mel{i+1}' for i in range(N_MELS)] + ['class'] # Define CSV header

# Output dataset as CSV
with open('dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(dataset)
print("\nSuccess! Output: dataset.csv") # Print after file is saved
        