import matplotlib.pyplot as plt
import torch

# mode = 'train'
# mode = 'val'
mode = 'test'

if mode == 'train':
    attention_weights = torch.load('./attention_weights_train.pt',
                                   map_location='cpu')
elif mode == 'val':
    attention_weights = torch.load('./attention_weights_val.pt',
                                   map_location='cpu')
else:
    attention_weights = torch.load('./attention_weights_test.pt',
                                   map_location='cpu')
batch_idx = 0
batch_size = len(attention_weights[0])
fig = plt.figure()
plt.tight_layout()
att_w = torch.from_numpy(attention_weights[0][batch_idx]).unsqueeze(0)
for at in attention_weights[1:]:
    att_w = torch.cat((att_w, torch.from_numpy(at[batch_idx]).unsqueeze(0)))
plt.imshow(att_w.detach().numpy())
plt.show()
