#!/usr/bin/env python
import argparse
import os.path
import wave
import numpy as np
import matplotlib.pyplot as plt

# Takes in directory containing 7 files: channel-1.wav ... channel-7.wav
# a timestamp, and a window length

# Parse input arguments
p = argparse.ArgumentParser()
p.add_argument("--start", type=float, default=0.0)
p.add_argument("--length", type=float, default=2.0)
p.add_argument("dir")
args = p.parse_args()

def read_channel(k):
    # Read the k-th wavefile and return
    fname = os.path.join(args.dir, 'channel-{:d}.wav'.format(k+1))
    fd = wave.open(fname)

    fs = fd.getframerate()
    width = fd.getsampwidth()

    if width == 2:
        datatype = np.int16
        maxval = 2**15
    elif width == 4:
        datatype = np.int32
        maxval = 2**31
    else:
        raise Exception("Can only handle 16 bit and 32 bit integer data")

    # Fast-forward to start value
    fd.setpos(round(args.start*fs))

    # Needs some error-checking added here
    frames = fd.readframes(round(args.length*fs))

    x = np.frombuffer(frames, dtype=datatype)

    # Turn into floats
    x = x/maxval
    return x

# Extract the data from the wavefiles
x = []
for k in range(7):
    x.append(read_channel(k))
    plt.plot(x[k])

plt.show()



    

