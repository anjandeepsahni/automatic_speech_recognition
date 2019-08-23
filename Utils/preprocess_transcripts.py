import os

import numpy as np

# Preprocess Data. Modify to make sentences.

DUMP_DATA_PATH = './../Data_Orig/Preprocessed'

SPEECH_DATA_PATH = './../Data_Orig'
# mode = 'train'

# SPEECH_DATA_PATH = './Data_Clean/Filtered_Dev'
mode = 'dev'

data = np.load(os.path.join(SPEECH_DATA_PATH,
                            '{}_transcripts.npy'.format(mode)),
               encoding='bytes')
for i, utt in enumerate(data):
    s = ""
    for w in utt:
        for c in w:
            s += chr(c)
        s += " "
    s = s[:-1]
    data[i] = s

np.save(os.path.join(DUMP_DATA_PATH, '{}_transcripts.npy'.format(mode)), data)
