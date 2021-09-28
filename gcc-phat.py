#!/usr/bin/env python
import argparse
import os.path
import wave
import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt

# Takes in directory containing 7 files: channel-1.wav ... channel-7.wav
# a timestamp, and a window length
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
    return x, fs

def delay_time(theta, pos, c):
    return -(pos[0]*np.cos(theta) + pos[1]*np.sin(theta))/c

def delay_sample(theta, pos, c, fs):
    return fs*delay_time(theta, pos, c)

if __name__ == "__main__":
    # Parse input arguments
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--length", type=float, default=2.0)
    p.add_argument("--speedofsound", type=float, default=343)
    p.add_argument("dir")
    args = p.parse_args()

    # Load the data from file
    data = []
    for k in range(7):
        x, fs = read_channel(k)
        data.append(x)
    data = np.array(data).T
    n_data = len(x)

    # Set up the camera array positions
    positions = 1e-3*np.array([
        [0, 0],
        [42.6, -0.5], 
        [21.5, -37.7],
        [-21.2, -37.7], 
        [-42.3, -0.8],
        [-21.2, 35.8],
        [21.4, 36.2],
        ])
    n_channels = len(positions)

    T = np.arange(n_data)/fs
    c = args.speedofsound


    # Set up the angular coordinates
    n_theta = 361; # angular resolution
    theta = np.linspace(0, 2*np.pi, 361)
    time_delays = np.zeros((n_channels, n_theta))
    sample_delays = np.zeros((n_channels, n_theta))

    for k in range(n_channels):
        x = positions[k]
        # Negative sign in front because we're dealing with delays
        time_delays[k] = delay_time(theta, x, c)
        sample_delays[k] = delay_sample(theta, x, c, fs)

    # Prepare the data
    u = data - np.mean(data, 0)
    # Zero pad the data (go whole hog to start with, can pare back later)
    u = np.vstack((u, np.zeros((100, n_channels))))
    U = []
    for k in range(n_channels):
        U.append(fft.rfft(u[:, k]))

    # Define which pairs of microphones we want to consider
    pairs = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], 
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], 
        [2, 3], [2, 4], [2, 5], [2, 6],
        [3, 4], [3, 5], [3, 6], 
        [4, 5], [4, 6],
        [5, 6]]

    theta = np.linspace(0, 2*np.pi, 1000)
    npairs = len(pairs)
    cc = []
    for k in range(npairs):
        i1, i2 = pairs[k]
        # GCC PHAT
        R = U[i1]*np.conj(U[i2])
        R = R/np.abs(R)
        foo = fft.irfft(R)
        idx = -delay_sample(theta, positions[i2], c, fs) + delay_sample(theta, positions[i1], c, fs)
        cc.append(foo[(idx.round()).astype(int)])

    cc = np.array(cc).T
    hm = 1/np.sum(1/np.abs(cc), 1)
    #plot(180/pi*theta, 1/sum(1/abs(cc), 1))
    plt.figure(1)
    plt.plot(T, data)
    plt.xlabel('t(sec)')

    plt.figure(2)
    plt.clf()
    plt.polar(theta, hm)
    
    plt.show()