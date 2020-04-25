import json
import torch
import random

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, f_name):
        with open(f_name) as f:
            self.data = json.load(f)
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.n_voc = 0 # count number of vocabularies

    def addWord(self):
        for voc in self.data:
            for i in voc['input']:
                self.add(i)
        
            self.add(voc['target'])
            self.n_voc += 1

        return self.word2index       
        
    def add(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def sample_pair(f_name):
    lang = Lang(f_name)
    index = lang.addWord()
    n = random.sample(range(lang.n_voc), 1)
    voc = lang.data[n[0]]
    input_tensor = []
    target_tensor = []
    for word in voc['input']:
        input_tensor.append(index[word])

    input_tensor.append(EOS_token)
    target_tensor.append(index[voc['target']])
    target_tensor.append(EOS_token)

    return (torch.tensor(input_tensor, dtype=torch.long).view(-1, 1), torch.tensor(target_tensor, dtype=torch.long).view(-1, 1))

'''   
p = sample_pair('train.json')
print(p[0])
print(p[1])
'''