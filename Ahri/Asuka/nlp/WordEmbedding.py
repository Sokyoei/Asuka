"""
Word Embedding 词嵌入
"""

import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torchtext.vocab import GloVe

WORDS = ["cat", "dog", "king", "queen"]


def main():
    glove = GloVe(name="6B", dim=300)
    embedding = nn.Embedding.from_pretrained(glove.vectors)
    print(f"{embedding.weight.shape=}")
    # WORDS 在词嵌入模型中的下标
    word_index = torch.tensor([glove.stoi[word] for word in WORDS])

    # print
    for word in WORDS:
        print(f"{word=}, {glove.stoi[word]=}")

    embedding_vectors: Tensor = embedding(word_index)
    vectors: NDArray = embedding_vectors.detach().numpy()
    print(f"{vectors.shape=}")


if __name__ == "__main__":
    main()
