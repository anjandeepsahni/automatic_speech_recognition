import os

import numpy as np

SPEECH_DATA_PATH = './../Data_Clean'
DUMP_DATA_PATH = './../Data_Clean/Filtered_Dev'

train_y = np.load(os.path.join(SPEECH_DATA_PATH, 'train_transcripts.npy'),
                  encoding='bytes')
dev_y = np.load(os.path.join(SPEECH_DATA_PATH, 'dev_transcripts.npy'),
                encoding='bytes')
dev_x = np.load(os.path.join(SPEECH_DATA_PATH, 'dev.npy'), encoding='bytes')
dup_list = []
for i in range(len(dev_y)):
    for j in range(len(train_y)):
        if np.array_equal(dev_y[i], train_y[j]):
            dup_list.append(i)
            break

dev_y_rev = np.delete(dev_y, dup_list)
dev_x_rev = np.delete(dev_x, dup_list)
assert (len(dev_y_rev) == len(dev_x_rev))

np.save(os.path.join(DUMP_DATA_PATH, 'dev.npy'), dev_x_rev)
np.save(os.path.join(DUMP_DATA_PATH, 'dev_transcripts.npy'), dev_y_rev)
