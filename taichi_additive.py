import taichi as ti
import pyaudio
import numpy as np
import time

ti.init(arch=ti.gpu)

samplerate = 48000
tau = 6.28318530717958647692

# Define LFO parameters
lfo_frequency_min = 55  # Minimum frequency of LFO
lfo_frequency_max = 995  # Maximum frequency of LFO
lfo_duration = 60  # Duration of LFO oscillation (in seconds)

# Calculate LFO parameters
lfo_samples = int(lfo_duration * samplerate)
lfo_frequency_range = lfo_frequency_max - lfo_frequency_min

shape = (128, 1024)  # 128 rows for each overtone
data = ti.field(dtype=ti.float32, shape=shape)
waveform_buffer = np.zeros(shape[1], dtype=np.float32)  # Buffer for the waveform data

# Initialize Taichi GUI
gui = ti.GUI("GPU Audio", res=(shape[1], shape[0]))  

@ti.kernel
def addOtones(t: float, last_phase: float, lfo_phase: float):
    # Calculate LFO frequency
    lfo_frequency = lfo_frequency_min + 0.5 * (ti.sin(tau * t / lfo_duration) + 1) * lfo_frequency_range

    for harmonic in range(128):
        for sample in range(shape[1]):
            n = harmonic + 1
            f0 = lfo_frequency
            
            if n * f0 < (samplerate // 2):
                phase = tau * n * f0 * (t + sample) / samplerate
            
                phase += last_phase - 2 * np.pi * int(last_phase / (2 * np.pi))
                data[harmonic, sample] = ti.sin(phase) / n

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
last_phase = 0.0
lfo_phase = 0.0

# Initialize phase of the last sample in the previous block
phase_last_block = 0.0

# Main loop
try:
    while gui.running and stream.is_active():
        start_time = time.time()
        addOtones(t, last_phase, lfo_phase)
        end_time = time.time()
        t += shape[1] / samplerate  # Update time
        
        # Update LFO phase
        lfo_phase += tau * lfo_frequency_min * (shape[1] / samplerate)
        
        # Calculate the phase angle of the waveform
        phase_angle = np.angle(waveform_buffer)
        
        # Normalize phase angle to [0, 2*pi] range
        phase_angle = phase_angle - 2 * np.pi * np.floor(phase_angle / (2 * np.pi))
        
        # Extract the phase direction (normalized between 0 and 1)
        phase_direction = phase_angle / (2 * np.pi)
        
        # Update phase of the last sample in the previous block
        phase_last_block = phase_angle[-1]
        
        # Apply phase adjustment to align current block with previous block
        for sample_index in range(shape[1]):
            phase_current_block = phase_angle[sample_index]
            phase_current_block += phase_last_block - 2 * np.pi * int(phase_last_block / (2 * np.pi))
            phase_angle[sample_index] = phase_current_block
        
        last_phase = phase_angle[-1]  # Update last phase
        
        # Synchronize Taichi kernel
        ti.sync()

        # Update the waveform buffer
        waveform_buffer = np.sum(data.to_numpy(), axis=0)
        waveform_buffer /= .002 + np.max(waveform_buffer)
        waveform_buffer *= .1
        # Normalize data for visualization
        img = 10 * data.to_numpy()  # Normalize to [0, 1]
        img_uint8 = (img * 255).astype(np.uint8)
        gui.set_image(img_uint8.T)
        gui.show()
        
        #print(f"Paint time: {end_time - start_time:.6f} s")

except KeyboardInterrupt:
    pass

finally:
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
