import os
import yaml
import csv
import wave
import numpy as np
import matplotlib.pyplot as plt
from yaml import CLoader as Loader

class MicrophoneArray:
    def __init__(self, coordinates):
        self.coordinates = coordinates

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
            data.append(x)
            self.U.append(fft.rfft(x))
        self.data = np.array(data).T
        self.n_frames = len(x)

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
        pos = np.dot(x, R.T) + np.array(config['array_positions'][key])
        arr[key] = MicrophoneArray(pos)
    return arr




# ------------------- MAIN SCRIPT STARTS HERE ---------------------#
# Start by reading in our configuration file.
with open('scripts/config.yml') as configfile:
    config = yaml.load(configfile, Loader = Loader)

# Load the microphone arrays
details, offset = load_experiment_details(config)
print(offset)

# Now construct the arrays
mic_arrays = construct_mic_arrays(config, details)

for key in mic_arrays:
    print(mic_arrays[key].coordinates)

num_channels = config['num_channels']
length = config['window']
dirs = {}
start = {}

aupfiles = {'c': 'path C', 'e': 'path E', 'w': 'path W'}
for key in ['c', 'e', 'w']:
    aupfile = os.path.join(config['data_dir'], details[aupfiles[key]])
    base, fname = os.path.split(aupfile)
    fname_base, ext = os.path.splitext(fname)
    dirs[key] = os.path.join(base, config['wav_dir'], fname_base)

start['c'] = float(details['time C']) - offset
start['e'] = float(details['time E']) - offset
start['w'] = float(details['time W']) - offset

for key in ['e', 'c', 'w']:
    m = mic_arrays[key]
    m.load_data(dirs[key], start[key], length, config['channel_prefix'], config['num_channels'])
    t = length * m.fs * np.arange(m.n_frames)
    plt.plot(t, mic_arrays[key].data[:,0])
#    def load_data(self, dir, start, length, prefix, num_channels):


print(details)
plt.show()