import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_reader import DataReader, PennDataset
from model import PennSkipGramModel


class PennTrainer:
    def __init__(self, input_file, output_file, emb_dimension=500, batch_size=32, window_size=5, iterations=5,
                 initial_lr=0.001, min_count=12):

        self.data = DataReader(input_file, min_count)
        dataset = PennDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.penn_skip_gram_model = PennSkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda: 
            self.penn_skip_gram_model.cuda()

    def train(self):

        for iteration in range(self.iterations):

            print("\n\n\nIteration: " + str(iteration + 1))
            optimizer = optim.SparseAdam(self.penn_skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v_l = sample_batched[1].to(self.device)
                    pos_v_r = sample_batched[2].to(self.device)
                    neg_v_l = sample_batched[3].to(self.device)
                    neg_v_r = sample_batched[4].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.penn_skip_gram_model.forward(pos_u, pos_v_l, pos_v_r, neg_v_l, neg_v_r)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))

            self.penn_skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


def lets_train(input_file, output_file):
    penn = PennTrainer(input_file, output_file)
    penn.train()
if __name__ == '__main__':
    lets_train(input_file="../data/google_analogy_sem.txt", output_file="../embeddings/out_penn_google_analogy_500dim_5w.vec")
    
