import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
    u_l_embedding: Embedding for center word to left side.
    u_r_embedding: Embedding for center word to right side.
    v_l_embedding: Embedding for left neighbor words.
    v_r_embedding: Embedding for right neighbor words.
"""

#Directional config
class PennSkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(PennSkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension

        self.u_l_embeddings = nn.Embedding(emb_size, emb_dimension//2, sparse=True)
        self.u_r_embeddings = nn.Embedding(emb_size, emb_dimension//2, sparse=True)
        self.v_l_embeddings = nn.Embedding(emb_size, emb_dimension//2, sparse=True)
        self.v_r_embeddings = nn.Embedding(emb_size, emb_dimension//2, sparse=True)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_l_embeddings.weight.data, -initrange, initrange)
        init.uniform_(self.u_r_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_l_embeddings.weight.data, 0)
        init.constant_(self.v_r_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v_l, pos_v_r, neg_v_l, neg_v_r):
        '''print("pos_u len ", pos_u.shape)
        print("pos_v_l len ", pos_v_l.shape)
        print("pos_v_r len ", pos_v_r.shape)
        print("neg_v_l len ", neg_v_l.shape)
        print("neg_v_r len ", neg_v_r.shape)
        print("embed_v_l len ", self.v_l_embeddings)
        print("embed_v_r len ", self.v_r_embeddings)
        print("embed_u_l len ", self.u_l_embeddings)
        print("embed_u_r len ", self.u_r_embeddings)'''
        
        emb_u_l = self.u_l_embeddings(pos_u)
        emb_u_r = self.u_r_embeddings(pos_u)
        emb_v_l = self.v_l_embeddings(pos_v_l)
        emb_v_r = self.v_r_embeddings(pos_v_r)
        emb_neg_v_l = self.v_l_embeddings(neg_v_l)
        emb_neg_v_r = self.v_r_embeddings(neg_v_r)

        score_l = torch.sum(torch.mul(emb_u_l, emb_v_l), dim=1)
        score_l = torch.clamp(score_l, max=10, min=-10)
        score_l = -F.logsigmoid(score_l)

        score_r = torch.sum(torch.mul(emb_u_r, emb_v_r), dim=1)
        score_r = torch.clamp(score_r, max=10, min=-10)
        score_r = -F.logsigmoid(score_r)

        neg_score_l = torch.bmm(emb_neg_v_l, emb_u_l.unsqueeze(2)).squeeze()
        neg_score_l = torch.clamp(neg_score_l, max=10, min=-10)
        neg_score_l = -torch.sum(F.logsigmoid(-neg_score_l), dim=1)

        neg_score_r = torch.bmm(emb_neg_v_r, emb_u_r.unsqueeze(2)).squeeze()
        neg_score_r = torch.clamp(neg_score_r, max=10, min=-10)
        neg_score_r = -torch.sum(F.logsigmoid(-neg_score_r), dim=1)
        return torch.mean(score_l + score_r + neg_score_l + neg_score_r)

    def save_embedding(self, id2word, file_name):
        embedding = (torch.cat((self.u_l_embeddings.weight, self.u_r_embeddings.weight), 1)).cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
