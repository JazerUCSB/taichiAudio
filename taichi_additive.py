import taichi as ti
import pyaudio
import numpy as np
import time
import threading  # Importing threading module

ti.init(arch=ti.gpu)

samplerate = 48000
tau = 6.28318530717958647692

lfo_frequency_min = 27.5
lfo_frequency_max = 27.5  
lfo_duration = 60  

lfo_samples = int(lfo_duration * samplerate)
lfo_frequency_range = lfo_frequency_max - lfo_frequency_min

shape = (128, 1024)  
data = ti.field(dtype=ti.float32, shape=shape)
waveform_buffer = np.zeros(shape[1], dtype=np.float32)  
waveform_buffer_2 = np.zeros(shape[1], dtype=np.float32)  

phase_offset = ti.field(dtype=ti.float32, shape=shape[0])

gui = ti.GUI("GPU Audio", res=(shape[1], shape[0]))

@ti.kernel
def addOtones(t: float, lfo_frequency: float):
    for harmonic in range(shape[0]):
        n = harmonic + 1
        f0 = lfo_frequency
        if n * f0 < ((samplerate // 2) - 1):
            phase_increment = (tau * n * f0) / samplerate
            for sample in range(shape[1]):
                phase = phase_offset[harmonic] + phase_increment * sample
                phase %= tau
                data[harmonic, sample] = ti.sin(phase) / n
            phase_offset[harmonic] = (phase_offset[harmonic] + phase_increment * shape[1]) % tau

p = pyaudio.PyAudio()
buffer_lock = threading.Lock()  # Lock for synchronizing access to buffers

def callback(in_data, frame_count, time_info, status):
    global waveform_buffer
    with buffer_lock:
        # Use the latest buffer data
        return (waveform_buffer.tobytes(), pyaudio.paContinue)

stream = p.open(format=p.get_format_from_width(4),
                channels=1,
                rate=samplerate,
                output=True,
                frames_per_buffer=shape[1],
                stream_callback=callback)

stream.start_stream()
t = 0

try:
    while stream.is_active():
        start_time = time.time()

        lfo_frequency = lfo_frequency_min + 0.5 * (np.sin(tau * (t / samplerate) / lfo_duration) + 1) * lfo_frequency_range

        addOtones(t, lfo_frequency)
        ti.sync()  # Ensure all kernels are done

        # Update the double buffer
        with buffer_lock:
            waveform_buffer_2 = np.sum(data.to_numpy(), axis=0)
            waveform_buffer_2 /= .002 + np.max(waveform_buffer_2)
            waveform_buffer_2 *= .1
            # Swap the buffers
            waveform_buffer, waveform_buffer_2 = waveform_buffer_2, waveform_buffer

        # Update GUI in the main thread
        img = 10 * np.abs(data.to_numpy())
        gui.set_image(img.T)
        gui.show()

        end_time = time.time()
        #print(f"Iteration time: {end_time - start_time:.6f} seconds")
        print(f"phase offset[0]: {phase_offset[2]:.6f}")

        t += 1

except KeyboardInterrupt:
    pass

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
