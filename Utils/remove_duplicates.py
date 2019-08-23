import os

import numpy as np

SPEECH_DATA_PATH = './../Data'
DUMP_DATA_PATH = './../Data_Clean'

mode = 'train'
# mode = 'dev'

data_x = np.load(os.path.join(SPEECH_DATA_PATH, '{}.npy'.format(mode)),
                 encoding='bytes')
data_y = np.load(os.path.join(SPEECH_DATA_PATH,
                              '{}_transcripts.npy'.format(mode)),
                 encoding='bytes')
dup_idx = np.load('{}_duplicates.npy'.format(mode), encoding='bytes')

assert (len(data_x) == len(data_y))
data_x_rev = np.delete(data_x, dup_idx)
data_y_rev = np.delete(data_y, dup_idx)
assert (len(data_x_rev) == len(data_y_rev))

np.save(os.path.join(DUMP_DATA_PATH, '{}.npy'.format(mode)), data_x_rev)
np.save(os.path.join(DUMP_DATA_PATH, '{}_transcripts.npy'.format(mode)),
        data_y_rev)
