import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""


class DiemSkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(DiemSkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v):
        '''print("pos_u len ", pos_u.shape)
        print("pos_v len ", pos_v.shape)
        print("neg_v len ", neg_v.shape)
        print("embed_v len ", self.v_embeddings)
        print("embed_u len ", self.u_embeddings)'''

        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        #emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        #neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        #neg_score = torch.clamp(neg_score, max=10, min=-10)
        #neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
