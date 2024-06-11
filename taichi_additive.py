import taichi as ti
import pyaudio
import numpy as np
import time

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

#phase_offset = ti.field(dtype=ti.float32, shape=shape[0])

gui = ti.GUI("GPU Audio", res=(shape[1], shape[0]))

@ti.kernel
def addOtones(t: float, lfo_frequency: float):
    for harmonic in range(shape[0]):
        n = harmonic + 1
        f0 = lfo_frequency
        if n * f0 < ((samplerate // 2) - 1):
            for sample in range(shape[1]):
                phase =  (tau * n * f0 * (sample + t)) / samplerate
                phase %= tau
                data[harmonic, sample] = ti.sin(phase) / n
           
            
p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, status):
    global waveform_buffer
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
    while gui.running and stream.is_active():
        
        lfo_frequency = lfo_frequency_min + 0.5 * (np.sin(tau * (t/samplerate) / lfo_duration) + 1) * lfo_frequency_range

        addOtones(t, lfo_frequency)
        
        t += shape[1] 
        
        ti.sync()

        waveform_buffer = np.sum(data.to_numpy(), axis=0)
        waveform_buffer /= .002 + np.max(waveform_buffer)
        waveform_buffer *= .1

        img = 10 * np.abs(data.to_numpy())
        gui.set_image(img.T)
        gui.show()

        #print(f"Paint time: {end_time - start_time:.6f} s")

except KeyboardInterrupt:
    pass

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
