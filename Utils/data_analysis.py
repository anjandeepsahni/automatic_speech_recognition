import os

import numpy as np

SPEECH_DATA_PATH = './../Data'
SPEECH_DATA_PATH = './Data_Clean'

mode = 'train'
# mode = 'dev'

data = np.load(os.path.join(SPEECH_DATA_PATH,
                            '{}_transcripts.npy'.format(mode)),
               encoding='bytes')

dup_list = []
duplicates = np.array([[0.1, 0.1]])
idx_list = list(np.arange(len(data)))
for i in range(len(data)):
    if i in dup_list:
        continue
    else:
        curr_dups = []
        for j in idx_list:
            if np.array_equal(data[j], data[i]) and (j != i):
                duplicates = np.vstack((duplicates, np.array([i, j])))
                curr_dups.append(j)
        curr_dups.append(i)
        dup_list.extend(curr_dups)
        for dup in curr_dups:
            idx_list.remove(dup)

duplicates = duplicates[1:, :]
assert (len(duplicates) == len(np.unique(duplicates[:, 1])))

per = len(duplicates) / len(data) * 100
print('Number of duplicate instances in %s data: %d/%d, %.2f %%' %
      (mode, len(duplicates), len(data), per))

remove_idx = np.unique(duplicates[:, 1])
np.save(mode + '_duplicates.npy', remove_idx)

# Test
if len(duplicates) > 1:
    idx = duplicates[0, 0]
    print('Testing with %s instance: %d' % (mode.upper(), int(idx)))
    sample_list = np.where(idx == duplicates[:, 0])[0]
    for s in sample_list:
        print(mode.upper(), 'Instance:', int(duplicates[s, 1]))
        print(data[int(duplicates[s, 1])])
