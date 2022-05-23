import sys
import torch
from more_itertools import collapse
from tqdm import tqdm
import numpy as np

# this file uses sentence transformer's sbert to embed MS MARCO

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v1').cuda()

# download ms marco
from datasets import load_dataset
dataset = load_dataset("ms_marco", 'v2.1')

# for every batch in ms marco, embed it using the model
query_embeddings = []
query_embeddings_no_dupe = []
answer_embeddings = []
answer_embeddings_no_dupe = []
passage_embeddings = []

bs = int(600)
N = int(1e5)
for i in tqdm(range(0, N, bs)):
    batch = dataset['validation'][i:i+bs]
    
    # get only the queries from the batch
    batch_queries = list(collapse([q for q in batch['query']]))

    # get only the answers from the batch
    batch_answers = list(collapse([a for a in batch['answers']]))

    # get only the passages from the batch
    batch_passages = [p['passage_text'] for p in batch['passages']]

    # embed. we need to convert the first two to a list so that we can expand below
    batch_queries = model.encode(batch_queries).tolist()
    batch_answers = model.encode(batch_answers).tolist()
    passage_embeddings.append(model.encode(list(collapse(batch_passages))))

    # append to no dupe
    query_embeddings_no_dupe.append(np.array(batch_queries))
    answer_embeddings_no_dupe.append(np.array(batch_answers))

    # expand batch queries and batch answers to the size of batch passages
    e_queries = []; [e_queries := e_queries + [q] * len(b) for q, b in zip(batch_queries, batch_passages)]
    e_answers = []; [e_answers := e_answers + [a] * len(b) for a, b in zip(batch_answers, batch_passages)]
    
    # append to the list of embeddings. convert back to a numpy array too
    query_embeddings.append(np.array(e_queries))
    answer_embeddings.append(np.array(e_answers))



# save embeddings to an npy
query_embeddings = np.concatenate(query_embeddings, axis=0)
answer_embeddings = np.concatenate(answer_embeddings, axis=0)
passage_embeddings = np.concatenate(passage_embeddings, axis=0)

query_embeddings_no_dupe = np.concatenate(query_embeddings_no_dupe, axis=0)
answer_embeddings_no_dupe = np.concatenate(answer_embeddings_no_dupe, axis=0)

np.save('ms_marco_query_embeddings_v1.npy', query_embeddings)
np.save('ms_marco_answer_embeddings_v1.npy', answer_embeddings)
np.save("ms_marco_passage_embeddings_v1.npy", passage_embeddings)

np.save('ms_marco_query_embeddings_no_dupe_v1.npy', query_embeddings_no_dupe)
np.save('ms_marco_answer_embeddings_no_dupe_v1.npy', answer_embeddings_no_dupe)
