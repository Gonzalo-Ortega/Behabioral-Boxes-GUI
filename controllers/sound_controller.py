"""Play a fixed frequency sound."""
from __future__ import division
import math
from pyaudio import PyAudio

# Configuration variables:
output_index = 3

try:
    from itertools import izip
except ImportError:
    izip = zip
    xrange = range


def configure(box):
    global output_index
    if box == 1:
        output_index = 3
    else:
        output_index = 1


# Previously lambda function s:
def sound(frequency, volume, sample_rate, sample):
    return int(volume * math.sin(2 * math.pi * frequency * sample / sample_rate) * 0x7f + 0x80)


def sine_tone(frequency, duration, volume=1, sample_rate=22050):
    print('[AUDIO] Tone')
    n_samples = int(sample_rate * duration)
    rest_frames = n_samples % sample_rate

    p = PyAudio()
    stream = p.open(format=p.get_format_from_width(1),  # 8bit
                    channels=1,  # mono
                    rate=sample_rate,
                    output=True,
                    output_device_index=output_index)

    samples = (sound(frequency, volume, sample_rate, sample) for sample in xrange(n_samples))
    # Write several samples at a time
    for buf in izip(*[samples] * sample_rate):
        stream.write(bytes(bytearray(buf)))

    # fill remainder of frame set with silence
    stream.write(b'\x80' * rest_frames)

    stream.stop_stream()
    stream.close()
    p.terminate()
