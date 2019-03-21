import numpy as np 

class Diem():
    def __init__(self, filename, M=5):
        self.filename = filename
        self.char_frequency = dict()
        self.char2id = dict()
        self.id2char = dict()
        self.vocab_size = 0
        self.char_embedding_size = 0
        self.char_embeddings = None
        self.M = M
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.embeddings = None
        for i, line in enumerate(open(self.filename, encoding="utf8")):
            if i == 0:
                line = line.split()
                self.vocab_size = int(line[0])
                self.char_embedding_size = int(line[1])
                self.char_embeddings = np.zeros((self.vocab_size, self.char_embedding_size))
                continue
            line = line.split(' ', 1)
            char = line[0]
            self.char2id[char] = i - 1
            self.id2char[i - 1] = char
            self.char_embeddings[i - 1, :] = np.fromstring(line[1], dtype=float, sep=' ')
    
    def embedding_for_char(self, c):
        ind = self.char2id[c]
        return self.char_embeddings[ind, :]

    def embedding_for_word(self, word):
        l = len(word)
        M = self.M
        C = self.char_embedding_size
        v = np.zeros(C * M)
        char_i = [self.embedding_for_char(char) for char in word]
        for i, c in enumerate(char_i): 
            s = M * i / l
            for m in range(M):
                d = np.power(1 - (np.abs(s - m)) / M, 2)
                v[m * C:(m + 1) * C] += d * c
        return v

    def learn_from_text(self, inputFileName, min_count=12):
        word_frequency = dict()
        for line in open(inputFileName, encoding="utf8"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1
        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1

        self.embeddings = np.zeros((wid, self.char_embedding_size * self.M))
        for i, w in self.id2word.items():
            self.embeddings[i, :] = self.embedding_for_word(w)

    def embedding_for(self, word):
        ind = self.word2id[word]
        return self.embeddings[ind, :]

if __name__ == '__main__':
    diem = Diem("../embeddings/out_diem.vec")
    print(diem.embedding_for_word("lol").shape)
    diem.learn_from_text("../data/enwik8_shorter_cleaned.txt")


