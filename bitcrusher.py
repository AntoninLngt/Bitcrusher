#thomas.hezard@mwm.ai

import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

# Parameters
# **********

# This must be an audio file readable by scipy.io.wavfile.read() function
input_file = "St Graal - Je t'emmènerai (Plan Séquence).wav"
#input_file = "Marimba.wav"
output_file = 'output.wav'


# Read input file
# ***************

fs, input_data = wavfile.read(input_file, mmap=True)
# convert to int
input_data = input_data.astype(np.int16)
# convert to mono if not already
if len(input_data.shape) > 1:
    input_data = np.mean(input_data, axis=1)
input_data = input_data.flatten()

# Audio data process
# ******************

output_data = np.zeros(len(input_data))

# *************************
# * PROCESSING COMES HERE *
# *************************

# bitcrusher

def bitcrusher(input_data, bitdepth, sample_rate_division):
    output_data = np.zeros_like(input_data)
    input_size = len(input_data)
    
    for i in range(0, input_size - sample_rate_division, sample_rate_division):
        chunk = input_data[i:i+sample_rate_division]  # Get a chunk of input data
        
        quantized_value = np.mean(chunk)  # Calculate the average value of the chunk
        
        quantized_value = round(quantized_value)  # Quantize the value
        
        # Reduce bit depth
        quantized_value >>= (16 - bitdepth)
        
        # Assign the quantized value to the output data
        output_data[i:i+sample_rate_division] = quantized_value
    
    return output_data

# Example usage
b = 14  # bitdepth
r = 10  # sample rate division

output_data = bitcrusher(input_data,b,r)


#low filter
output_low_data = np.zeros(len(input_data))

def low_pass_filter(input_data, alpha):
    
    for i in range(len(input_data)):
        if i == 0:
            output_low_data[i] = input_data[i]  # Initial value
        else:
            output_low_data[i] = alpha * input_data[i] + alpha * output_low_data[i - 1]
    
    return output_low_data


alpha = 0.7
output_low_data = low_pass_filter(output_data,alpha)

# Save output
# ***********
output_data_normalized = 0.99 * output_low_data / max(abs(output_low_data))
wavfile.write(output_file , int(fs),(output_data_normalized * np.iinfo(np.int16).max).astype(np.int16))



# Normalize input data for display
input_data_normalized = 0.99 * input_data / max(abs(input_data))


# Plot input audio signal
plt.figure(figsize=(10, 6))

# Define the duration to plot (in seconds)
duration_to_plot = 20

# Calculate the number of samples corresponding to the specified duration
num_samples_to_plot = int(duration_to_plot * fs)

# Plot input audio signal (first 20 seconds)
plt.plot(np.arange(num_samples_to_plot) / fs, input_data_normalized[:num_samples_to_plot], label='Input Audio', color='blue')

# Plot output audio signals after bitcrushing (first 20 seconds)
output_data_normalized_filterless = 0.99 * output_data / max(abs(output_data))
plt.plot(np.arange(num_samples_to_plot) / fs, output_data_normalized_filterless[:num_samples_to_plot], label='Bitcrushed', color='red')

plt.plot(np.arange(num_samples_to_plot) / fs, output_data_normalized[:num_samples_to_plot], label='Bitcrushed+lowfilter', color='green')

plt.title('Comparison of Input and Bitcrushed Output (First 20 Seconds)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Specify the legend location explicitly
plt.legend(loc='best')

plt.grid(True)
plt.tight_layout()
plt.show()

