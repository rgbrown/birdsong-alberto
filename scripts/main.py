import os
import yaml
import csv
import wave
import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
from yaml import CLoader as Loader

class MicrophoneArray:
    def __init__(self, coordinates, centre, c_sound=343):
        self.coordinates = coordinates
        self.centre = centre
        self.c_sound = c_sound

    def load_data(self, dir, start, length, prefix, num_channels):
        data = []
        self.U = []
        for k in range(num_channels):
            fname = os.path.join(dir, '{:s}{:d}.wav'.format(prefix, k+1))
            fd = wave.open(fname)
            if k == 0:
                self.fs = fd.getframerate()
            width = fd.getsampwidth()

            if width == 2:
                datatype = np.int16
                maxval = 2**15
            elif width == 4:
                datatype = np.int32
                maxval = 2**31
            else:
                raise Exception("Can only handle 16 bit and 32 bit integer data")
        
            #Fast-forward to start value
            fd.setpos(round(start*self.fs))

            frames = fd.readframes(round(length*self.fs))
            x = np.frombuffer(frames, dtype=datatype)/maxval

            # Process data
            # Remove mean
            x -= np.mean(x)

            # Zero pad
            x = np.concatenate((x, np.zeros(100)))
            
            data.append(x)
            self.U.append(fft.rfft(x))
        self.data = np.array(data).T
        self.n_frames = len(x)

    def delay_time(self, theta, pos):
        x = [pos[0] - self.centre[0], pos[1] - self.centre[1]]
        return -(x[0]*np.cos(theta) + x[1]*np.sin(theta))/self.c_sound

    def delay_sample(self, theta, pos):
        return self.fs*self.delay_time(theta, pos)

    def gcc_phat(self, pairs, pos):
        # Calculate angles of arrival
        x = pos
        theta = np.arctan2(x[1] - self.centre[1], x[0] - self.centre[0])
    
        npairs = len(pairs)
        cc = []
        for k in range(npairs):
            i1, i2 = pairs[k]
            R = self.U[i1]*np.conj(self.U[i2])
            R = R/np.abs(R)
            foo = fft.irfft(R)
            d1 = self.delay_sample(theta, self.coordinates[i1]) 
            d2 = -self.delay_sample(theta, self.coordinates[i2])
            idx = d1 + d2
            cc.append(foo[(idx.round()).astype(int)])
        return np.array(cc).T


# This is going to be the do-everything script. 
def load_experiment_details(config):
    with open(config['experiment_file']) as csvfile:
        reader = csv.DictReader(csvfile)
        rows = []
        for row in reader:
            rows.append(row)

    with open(config['sound_offsets_file']) as csvfile:
        reader = csv.reader(csvfile)
        offsets = []
        header = next(reader)
        for row in reader:
            offsets.append(float(row[0]))
        offset = offsets[config['sound_id']]
    
    return rows[config['experiment_id']], offset

def construct_mic_arrays(config, details):
    arr = {}
    x = np.array(config['coordinates'])
    for key in ['w', 'c', 'e']:
        theta = config['array_orientations'][key]*np.pi/180
        c = np.cos(theta)
        s = np.sin(theta)

        # Fix the orientation
        R = np.array([[c, -s], [s, c]])
        centre = np.array(config['array_positions'][key])
        pos = np.dot(x, R) + centre # Using the inverse of R, but multiplying on right, means R is what we have
        arr[key] = MicrophoneArray(pos, centre, c_sound=config['speed_of_sound'])
    return arr



# ------------------- MAIN SCRIPT STARTS HERE ---------------------#
# Start by reading in our configuration file.
with open('scripts/config.yml') as configfile:
    config = yaml.load(configfile, Loader = Loader)

# Load the microphone arrays
details, offset = load_experiment_details(config)

# Now construct the arrays
mic_arrays = construct_mic_arrays(config, details)

num_channels = config['num_channels']
length = config['window']
dirs = {}
start = {}

aupfiles = {'c': 'path C', 'e': 'path E', 'w': 'path W'}
times = {'c': 'time C', 'e': 'time E', 'w': 'time W'}
for key in ['c', 'e', 'w']:
    aupfile = os.path.join(config['data_dir'], details[aupfiles[key]])
    base, fname = os.path.split(aupfile)
    fname_base, ext = os.path.splitext(fname)
    dirs[key] = os.path.join(base, config['wav_dir'], fname_base)
    start[key] = float(details[times[key]]) - offset

plt.figure(1)
for key in ['e', 'c', 'w']:
    m = mic_arrays[key]
    m.load_data(dirs[key], start[key], length, config['channel_prefix'], config['num_channels'])
    t = 1/m.fs * np.arange(m.n_frames)
    plt.plot(t, mic_arrays[key].data[:,0])
#    def load_data(self, dir, start, length, prefix, num_channels):

m = mic_arrays['c']
x = np.arange(-75, 76, 1)
y = np.arange(-50, 51, 1)
X, Y = np.meshgrid(x, y)
cc = {}
for key in ['e', 'c', 'w']:
    m = mic_arrays[key]
    cc[key]  = m.gcc_phat(config['pairs'], [X, Y])

cc = np.concatenate((cc['c'], cc['e'], cc['w']), axis=2)
hm = 1/np.sum(1/np.abs(cc), 2)
plt.figure(2)
plt.imshow(np.sum(cc, 2).T, origin='lower', extent=[-75.5, 75.5, -50.5, 50.5])

print(details)
plt.show()