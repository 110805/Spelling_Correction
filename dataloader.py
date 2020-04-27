import json
import torch

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self):
        self.data = {}
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.n_groups = 0 # count number of vocabularies

    def addWord(self, voc):
        self.data = voc
        for data in voc:
            for i in data['input']:
                self.add(i)
        
            self.add(data['target'])
            self.n_groups += 1

        return        
        
    def add(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def sample_pair(lang, i):
    voc = lang.data[i]
    input_tensor = []
    target_tensor = []
    for word in voc['input']:
        input_tensor.append(lang.word2index[word])

    input_tensor.append(EOS_token)
    target_tensor.append(lang.word2index[voc['target']])
    target_tensor.append(EOS_token)

    return (torch.tensor(input_tensor, dtype=torch.long).view(-1, 1), torch.tensor(target_tensor, dtype=torch.long).view(-1, 1))

'''
with open('train.json') as f:
    voc = json.load(f)

lang = Lang()
lang.addWord(voc)
print(sample_pair(lang))
'''
