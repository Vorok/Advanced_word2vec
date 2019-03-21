import numpy as np
import torch
from torch.utils.data import Dataset

np.random.seed(12345)


class DiemDataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFileName, min_count):

        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.char2id = dict()
        self.id2char = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.char_frequency = dict()

        self.inputFileName = inputFileName
        self.read_words(min_count)

    def read_words(self, min_count):
        char_frequency = dict()
        for line in open(self.inputFileName, encoding="utf8"):
            line = line.split()
            if len(line) > 0:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        for char in word:
                            self.token_count += 1
                            char_frequency[char] = char_frequency.get(char, 0) + 1

                            if self.token_count % 1000000 == 0:
                                print("Read " + str(int(self.token_count / 1000000)) + "M chars.")

        wid = 0
        for w, c in char_frequency.items():
            if c < min_count:
                continue
            self.char2id[w] = wid
            self.id2char[wid] = w
            self.char_frequency[wid] = c
            wid += 1
        print("Total embeddings: " + str(len(self.char2id)))



# -----------------------------------------------------------------------------------------------------------------

class DiemCharDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, encoding="utf8")

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 0:
                word = line.split()
                if len(word) > 0:
                    for chars in word:
                        char_ids = [self.data.char2id[c] for c in chars]
                        boundary = np.random.randint(1, self.window_size)
                        return [(u, v) for i, u in enumerate(char_ids) for j, v in
                                enumerate(char_ids[max(i - boundary, 0):i + boundary]) if u != v]

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v)
