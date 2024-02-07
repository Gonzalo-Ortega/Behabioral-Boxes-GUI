"""Play a fixed frequency sound."""
from __future__ import division
import math
from pyaudio import PyAudio  # sudo apt-get install python{,3}-pyaudio

# Configuration variables:
output_index = 3


try:
    from itertools import izip
except ImportError:  # Python 3
    izip = zip
    xrange = range


def configure(box):
    global output_index
    if box == 1:
        output_index = 3
    else:
        output_index = 1


def sine_tone(frequency, duration, volume=1, sample_rate=22050):
    n_samples = int(sample_rate * duration)
    restframes = n_samples % sample_rate

    p = PyAudio()
    stream = p.open(format=p.get_format_from_width(1),  # 8bit
                    channels=1,  # mono
                    rate=sample_rate,
                    output=True,
                    output_device_index=output_index)
    s = lambda t: volume * math.sin(2 * math.pi * frequency * t / sample_rate)
    samples = (int(s(t) * 0x7f + 0x80) for t in xrange(n_samples))
    for buf in izip(*[samples] * sample_rate):  # write several samples at a time
        stream.write(bytes(bytearray(buf)))

    # fill remainder of frameset with silence
    stream.write(b'\x80' * restframes)

    stream.stop_stream()
    stream.close()
    p.terminate()
