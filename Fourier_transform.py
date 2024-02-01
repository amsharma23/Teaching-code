#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:17:54 2024

@author: amansharma"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
#from ipywidgets import interactive, FloatSlider

#%% 1D Fourier Transform
def gaussian_waveform(x, mean, std_dev):
    return np.exp(-0.5 * ((x - mean) / std_dev)**2)

def update_gaussian_plot(std_dev):
    x = np.linspace(-1, 1, 10000)
    x_s = np.linspace(-1,1, 10000)
    y = gaussian_waveform(x, 0, std_dev)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, y)
    plt.title('Original Signal: SD-'+str(std_dev))
    
    fft_result = fftshift(np.abs(fft(y)))
    
    plt.subplot(1, 2, 2)
    plt.scatter(x_s, (fft_result)/np.sqrt(len(fft_result)),s=0.1)
    plt.title('Fourier Transform')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.show()

# Get standard deviation from user input
std_dev = float(0.00002)
update_gaussian_plot(std_dev)





#%% 2D Fourier Transform
def plot_spectrum(fft_result):
    magnitude_spectrum = np.abs(fft_result)
    plt.imshow(np.log1p(magnitude_spectrum), cmap='gray')
    plt.title('2D Fourier Transform')
    
    # Add colorbar with label
    cbar = plt.colorbar()
    cbar.set_label('Log Magnitude')

    plt.show()

# Example usage:
# Create a simple 2D image
x = np.linspace(-5, 5, 256)
y = np.linspace(-5, 5, 256)
X, Y = np.meshgrid(x, y)
image = np.sin(np.sqrt(X**2 + Y**2))
plt.imshow((image), cmap='gray')
plt.title('Image')

# Add colorbar with label
plt.colorbar()
plt.show()
# Perform 2D Fourier transform using scipy.fft
fft_result = fftshift(fft2(image))

# Plot the magnitude spectrum with colorbar label
plot_spectrum(fft_result)
