from nltk.corpus import wordnet as wn
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

data = """The cat and her kittens
They put on their mittens,
To eat a Christmas pie.
The poor little kittens
They lost their mittens,
And then they began to cry."""


data2 = """O mother dear, we sadly fear
We cannot go to-day,
For we have lost our mittens."
"If it be so, ye shall not go,
For ye are naughty kittens."""



corpus = data.lower().split("\n")
corpus2 = data2.lower().split("\n")

t = Tokenizer(filters='!"#$%&()*+,-./;<=>?@[\\]^_`{|}~\t\n', lower=True)
t.fit_on_texts(corpus)
t.fit_on_texts(corpus2)

print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)

# integer encode documents
# encoded_docs = t.texts_to_matrix(corpus, mode='count')
# encoded_docs2 = t.texts_to_matrix(corpus2, mode='count')
# print(encoded_docs)

encoded = t.texts_to_sequences(corput)

input_sequences = []

for line in corpus:
    token_list = t.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)