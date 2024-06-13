import taichi as ti
import pyaudio
import numpy as np
import threading
import time
import multiprocessing.shared_memory as shm

ti.init(arch=ti.gpu)

samplerate = 48000
tau = np.pi * 2

lfo_frequency_min = 44
lfo_frequency_max = 44
lfo_duration = 100

lfo_samples = int(lfo_duration * samplerate)
lfo_frequency_range = lfo_frequency_max - lfo_frequency_min

shape = (128, 48000)
data = ti.field(dtype=ti.float32, shape=shape)
phase_offset = ti.field(dtype=ti.float32, shape=shape[0])
summed_wave = ti.field(dtype=ti.float32, shape=shape[1])

@ti.kernel
def addOtones(t: float, lfo_frequency: float):
    for harmonic in range(shape[0]):
        n = harmonic + 1
        f0 = lfo_frequency
        if n * f0 < ((samplerate // 2) - 1):
            phase_increment = (ti.math.pi * 2 * n * f0) / samplerate
            for sample in range(shape[1]):
                phase = phase_increment * sample + phase_offset[harmonic]
                phase = ti.math.mod(phase, ti.math.pi * 2)
                data[harmonic, sample] = ti.sin(phase) / n
            phase_offset[harmonic] = ti.math.mod(phase_offset[harmonic] + phase_increment * shape[1], tau)

@ti.kernel
def sumOtones():
    for col in range(shape[1]):
        summed_wave[col] = 0
        for row in range(shape[0]):
            summed_wave[col] += data[row, col]

# Size in bytes of a float32 element
element_size = 4

# Initialize shared memory for summed_wave
summed_wave_shm = shm.SharedMemory(create=True, size=shape[1] * element_size)
summed_wave_np = np.ndarray(summed_wave.shape, dtype=np.float32, buffer=summed_wave_shm.buf)

p = pyaudio.PyAudio()
buffer_lock = threading.Lock()  # Lock for synchronizing access to buffers

def callback(in_data, frame_count, time_info, status):
    with buffer_lock:
        return (summed_wave_np.tobytes(), pyaudio.paContinue)

stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=samplerate,
                output=True,
                frames_per_buffer=shape[1],
                stream_callback=callback)

stream.start_stream()
t = 0

gui = ti.GUI("GPU Audio", res=(shape[1], shape[0]))

try:
    while stream.is_active():
        lfo_frequency = lfo_frequency_min + 0.5 * (np.sin(tau * (t / samplerate) / lfo_duration) + 1) * lfo_frequency_range
        addOtones(t, lfo_frequency)
        sumOtones()
        ti.sync()

        # Copy summed_wave to shared memory
        summed_wave_np[:] = summed_wave.to_numpy()

        img = 10 * np.abs(data.to_numpy())
        gui.set_image(img.T)
        gui.show()

        t += shape[1]

except KeyboardInterrupt:
    pass

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    summed_wave_shm.close()
    summed_wave_shm.unlink()
