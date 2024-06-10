import taichi as ti
import pyaudio
import numpy as np
import time

ti.init(arch=ti.gpu)

samplerate = 48000
tau = 6.28318530717958647692

# Define LFO parameters
lfo_frequency_min = 27.5  # Minimum frequency of LFO
lfo_frequency_max = 227.5  # Maximum frequency of LFO
lfo_duration = 60  # Duration of LFO oscillation (in seconds)

# Calculate LFO parameters
lfo_samples = int(lfo_duration * samplerate)
lfo_frequency_range = lfo_frequency_max - lfo_frequency_min

shape = (128, 1024)  # 128 rows for each overtone
data = ti.field(dtype=ti.float32, shape=shape)
waveform_buffer = np.zeros(shape[1], dtype=np.float32)  # Buffer for the waveform data

# Store the phase for each harmonic
phase_offset = ti.field(dtype=ti.float32, shape=shape[0])


# Initialize Taichi GUI
gui = ti.GUI("GPU Audio", res=(shape[1], shape[0]))

@ti.kernel
def addOtones(lfo_frequency: float):
    for harmonic in range(128):
        n = harmonic + 1
        f0 = lfo_frequency
        if n * f0 < ((samplerate // 2) - 1):
            for sample in range(shape[1]):
                phase = (tau * n * f0 * (sample + phase_offset[harmonic])) / samplerate
                if phase>tau:
                    phase -= tau
                data[harmonic, sample] = ti.sin(phase) / n
            phase_offset[harmonic] = (phase_offset[harmonic] + shape[1]) % shape[1]     
            

# Initialize PyAudio
p = pyaudio.PyAudio()


# Define the stream callback function
def callback(in_data, frame_count, time_info, status):
    global waveform_buffer
    # Return the precomputed waveform buffer
    return (waveform_buffer.tobytes(), pyaudio.paContinue)

# Create a PyAudio stream in non-blocking mode
stream = p.open(format=p.get_format_from_width(4),
                channels=1,
                rate=samplerate,
                output=True,
                frames_per_buffer=shape[1],
                stream_callback=callback)

# Start the stream
stream.start_stream()

# Initialize time variable
t = 0

# Main loop
try:
    while gui.running and stream.is_active():
        

        # Calculate LFO frequency
        lfo_frequency = lfo_frequency_min + 0.5 * (np.sin(tau * t / lfo_duration) + 1) * lfo_frequency_range

        addOtones(lfo_frequency)
        
        t += shape[1] / samplerate # Update time
        
        # Synchronize Taichi kernel
        ti.sync()

        # Update the waveform buffer
        waveform_buffer = np.sum(data.to_numpy(), axis=0)
        waveform_buffer /= .002 + np.max(waveform_buffer)
        waveform_buffer *= .1
        
        # Normalize data for visualization
        img = 10 * np.abs(data.to_numpy())
        gui.set_image(img.T)
        gui.show()

        #print(f"Paint time: {end_time - start_time:.6f} s")

except KeyboardInterrupt:
    pass

finally:
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
