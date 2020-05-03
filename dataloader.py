import json
import torch

SOS_token = 0
EOS_token = 1

def collect_data():
    Data = []
    with open('train.json') as f:
        Voc = json.load(f)

    for voc in Voc:
        for i in range(len(voc['input'])):
            group = []
            group.append(voc['input'][i])
            group.append(voc['target'])
            Data.append(group)

    return Data
    
def sample_pair(i):
    Data = collect_data()
    input_tensor = []
    target_tensor = []
    
    for input_char in Data[i][0]:
        input_tensor.append(ord(input_char)-95)
    
    for target_char in Data[i][1]:
        target_tensor.append(ord(target_char)-95)

    return (torch.tensor(input_tensor, dtype=torch.long).view(-1, 1), torch.tensor(target_tensor, dtype=torch.long).view(-1, 1))

i, j = sample_pair(12924)
print(i)
print(j)