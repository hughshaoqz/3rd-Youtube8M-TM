from bert_embedding import BertEmbedding
import pandas as pd
import numpy as np

bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')

df = pd.read_csv('../inputs/data/vocabulary_all.csv')

vocab_index = df.Index.values
vocab_size = len(vocab_index)
bert_abstract = []
for sentence in df.Name.values:
    bert_abstract.append(sentence)

result = {}
for i, sentence in enumerate(bert_abstract):
    print(i, sentence)
    try:
        tokens = sentence.split('\n')
    except Exception as e:
        result[vocab_index[i]] = np.zeros((vocab_size, 1024))
        print(e)
    result[vocab_index[i]] = np.mean(bert_embedding(tokens)[0][1], 0)

result_df = pd.DataFrame(result).T
result_df.to_csv('../inputs/data/vocabulary_all_embedding.csv')