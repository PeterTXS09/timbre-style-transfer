# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:57:16 2020

@author: Javier
"""


import numpy as np
import scipy.io.wavfile
import sounddevice as sd
import argparse
import librosa
import matplotlib
import os
from data import files_within, init_directory

def remove_hf(mag):
    return mag[0:int(mag.shape[0]/2), :]

def slice_first_dim(array, slice_size):
    n_sections = int(np.floor(array.shape[1]/slice_size))
    has_last_mag = n_sections*slice_size < array.shape[1]

    last_mag = np.zeros(shape=(1, array.shape[0], slice_size, array.shape[2]))
    last_mag[:,:,:array.shape[1]-(n_sections*slice_size),:] = array[:,n_sections*int(slice_size):,:]
    
    if(n_sections > 0):
        array = np.expand_dims(array, axis=0)
        sliced = np.split(array[:,:,0:n_sections*slice_size,:], n_sections, axis=2)
        sliced = np.concatenate(sliced, axis=0)
        if(has_last_mag): # Check for reminder
            sliced = np.concatenate([sliced, last_mag], axis=0)
    else:
        sliced = last_mag
    return sliced

def forward_transform(audio, nfft=1024, normalize=True, crop_hf=True):
    window = np.hanning(nfft)
    S = librosa.stft(audio, n_fft=nfft, hop_length=int(nfft/2), window=window)
    mag, phase = np.abs(S), np.angle(S)
    if(crop_hf):
        mag = remove_hf(mag)
    if(normalize):
        mag = 2 * mag / np.sum(window)
    return mag, phase

def amplitude_to_db(mag, amin=1/(2**16), normalize=True):
    mag_db = 20*np.log1p(mag/amin)
    if(normalize):
        mag_db /= 20*np.log1p(1/amin)
    return mag_db

def slice_magnitude(mag, slice_size):
    magnitudes = np.stack([mag], axis=2)
    return slice_first_dim(magnitudes, slice_size)

def wavToImg(filename,origin_path,target_path):
    samplerate, data = scipy.io.wavfile.read(origin_path+'/'+filename);    
    datnorm = data * (1/abs(data).max())
    mag_input, phase = forward_transform(datnorm)                       # Short Time Fourier Transform
    mag_input = amplitude_to_db(mag_input)                              # convierte a amplitud db
    test_input = slice_magnitude(mag_input, mag_input.shape[0])         # corta en pedazos de 256x256
    test_input = (test_input * 2) - 1                                   # escala de [0,1] a [-1,1]
    base = os.path.splitext(filename)[0]+'_'
    for slice in range(test_input.shape[0]):
        matplotlib.image.imsave(os.path.join(target_path,base+str(slice+1).zfill(3)+'.png'),
                                test_input[slice,:,:,0],
                                cmap='nipy_spectral',
                                origin='lower')
#        matplotlib.pyplot.imshow(test_input[slice,:,:,0],cmap='nipy_spectral',origin='lower')   # muestra imagen
#        matplotlib.pyplot.show()

def playWav(filename):
    samplerate, data = scipy.io.wavfile.read(filename);
    sd.play(data, samplerate);
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--origin_path', required=True)
    ap.add_argument('--target_path', required=True)
    args = ap.parse_args()

    wavfiles = list(files_within(args.origin_path, '*.wav'))
    for wavv in wavfiles:
        _, seq_name = os.path.split(wavv)
        print("seq_name: ",seq_name)
        print('\n')
        print('Archivo: ',seq_name)
        print('\n')
        wavToImg(seq_name,args.origin_path,args.target_path)
    
    
    
