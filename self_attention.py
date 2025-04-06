import numpy as np
from math import sqrt


class SelfAttention:
    def __init__(self, d_model, heads):
        self.d_model = d_model
        self.heads = heads

    def get_embeddings(self, n):
        np.random.seed(42)
        embeddings = np.random.randn(n, self.d_model)

        for pos in range(n):
            for i in range(self.d_model):
                if i % 2 == 0:
                    sin_emb = np.sin(pos/1000**(2*i/self.d_model))
                    embeddings[pos, i] += sin_emb
                else:
                    cos_emb = np.cos(pos/1000**(2*i/self.d_model))
                    embeddings[pos, i] += cos_emb
        return embeddings

    def get_q_k_v(self, embeddings):
        q = np.random.randn(embeddings.shape[0], self.heads)
        k = np.random.randn(embeddings.shape[0], self.heads)
        v = np.random.randn(embeddings.shape[0], self.heads)

        return q, k, v

    def get_dot_product(self, q, k):
        return np.dot(q, k.T)

    def get_scaled_dot(self, dot_product):
        return np.divide(dot_product, sqrt(self.heads))

    def get_softmax(self, scaled_dot):
        e_x = np.exp(scaled_dot - np.max(scaled_dot))
        return e_x / np.sum(e_x)

    def get_attention_output(self, q, k, v):
        dot_product = self.get_dot_product(q, k)
        scaled_dot = self.get_scaled_dot(dot_product)
        softmax = self.get_softmax(scaled_dot)
        return np.dot(softmax, v)


self_attention = SelfAttention(4, 2)

n = 5
embeddings = self_attention.get_embeddings(n)

q, k, v = self_attention.get_q_k_v(embeddings)

outputs = self_attention.get_attention_output(q, k, v)
print("embeddings: \n", embeddings)
print("q: \n", q)
print("k: \n", k)
print("v: \n", v)
print("embeddings after self attention: \n", outputs)
