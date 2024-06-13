import taichi as ti
import pyaudio
import numpy as np

ti.init(arch=ti.gpu)

samplerate = 48000
tau = np.pi * 2

f0 = 100
shape = (1, 4800)
data = ti.field(dtype=ti.float32, shape=shape)
phase_offset = ti.field(dtype=ti.float32, shape=shape[0])
summed_wave = ti.field(dtype=ti.float32, shape=shape[1])

@ti.kernel
def addOtones(t: float):
    for harmonic in range(shape[0]):
        n = harmonic + 1  
        if n * f0 < ((samplerate // 2) - 1):
            phase_increment = (tau * n * f0) / samplerate
            for sample in range(shape[1]):
                phase = phase_increment * sample + phase_offset[harmonic]
                phase %= tau 
                data[harmonic, sample] = ti.sin(phase) / n       
            phase_offset[harmonic] = (phase_increment * shape[1] + phase_offset[harmonic]) % tau 
@ti.kernel
def sumOtones():
    for col in range(shape[1]):
        summed_wave[col] = 0
        for row in range(shape[0]):
            summed_wave[col] += data[row, col]

@ti.kernel
def sumPhase():
    for harmonic in range(shape[0]):
        n = harmonic + 1
        if n * f0 <((samplerate//2) - 1):
           phase_increment = (tau * n * f0) / samplerate
           phase_offset[harmonic] = ti.math.mod(phase_offset[harmonic] + phase_increment * shape[1], tau)

p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, status):
        return (summed_wave.to_numpy(), pyaudio.paContinue)

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
        
        addOtones(t)
        sumOtones()
        #sumPhase()
        ti.sync()

        img = 10 * np.abs(data.to_numpy())
        gui.set_image(img.T)
        gui.show()

        t += shape[1]
        #print(phase_offset.to_numpy()[:10])
except KeyboardInterrupt:
    pass

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

