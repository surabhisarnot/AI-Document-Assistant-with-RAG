import numpy as np
from ai_document_assistant_with_rag.embed import embed_texts

texts = [
    "The cat sat on the mat.",
    "A kitten is resting on a rug.",
    "Quarterly revenue grew by 12%.",
]
vecs = np.array(embed_texts(texts))
print("vector length:", vecs.shape[1])
print("first 5 numbers:", vecs[0][:5])


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print("cat vs kitten: ", round(cos(vecs[0], vecs[1]), 3))
print("cat vs revenue:", round(cos(vecs[0], vecs[2]), 3))