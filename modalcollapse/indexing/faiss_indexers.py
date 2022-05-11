# taken from: https://github.com/facebookresearch/DPR/blob/main/dpr/indexer/faiss_indexers.py

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 FAISS-based index components for dense retriever
"""

import faiss
import logging
import numpy as np
import os
import pickle

from typing import List, Tuple

logger = logging.getLogger()


class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        """
        Initialize the indexer.
        :param buffer_size: size of the buffer to use for indexing
        """
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None
        self.centroids = None
        self.copy_of_points = list()

    def init_index(self, vector_sz: int):
        """
        Initialize the index.
        :param vector_sz: size of the vectors to index
        """
        raise NotImplementedError

    def get_centroids(self):
        """
        Compute the centroids from the index.
        """
        self.centroids = self.index.quantizer.reconstruct_n(0, self.index.nlist)


    def index_data(self, data: List[Tuple[object, np.array]]):
        """
        Index the given data.
        :param data: list of tuples of (id, vector)
        """
        raise NotImplementedError

    def train(self, vectors: np.array):
        """
        Train the index.
        :param vectors: vectors to train the index with
        """
        self.index.train(vectors)

    def get_index_name(self):
        """
        Get the name of the index.
        """
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        """
        Search for the top_docs nearest neighbors of the given query vectors.
        :param query_vectors: query vectors
        :param top_docs: number of nearest neighbors to return
        :return: list of tuples of (id, distance)
        """
        raise NotImplementedError

    def set_copy_of_points(self, copy_of_points: List):
        """
        Set the copy of points to use for indexing.
        """
        self.copy_of_points = copy_of_points

    def serialize(self, file: str):
        """
        Serialize the index to disk.
        :param file: path to the file to write to
        """
        logger.info("Serializing index to %s", file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + ".index.dpr"
            meta_file = file + ".index_meta.dpr"

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode="wb") as f:
            pickle.dump(self.index_id_to_db_id, f)

    def get_files(self, path: str):
        """
        Get the index and meta files for the given path.
        :param path: path to the index
        """
        if os.path.isdir(path):
            index_file = os.path.join(path, "index.dpr")
            meta_file = os.path.join(path, "index_meta.dpr")
        else:
            index_file = path + ".{}.dpr".format(self.get_index_name())
            meta_file = path + ".{}_meta.dpr".format(self.get_index_name())
        return index_file, meta_file

    def index_exists(self, path: str):
        """
        Check if the index exists at the given path.
        :param path: path to the index
        """
        index_file, meta_file = self.get_files(path)
        return os.path.isfile(index_file) and os.path.isfile(meta_file)

    def deserialize(self, path: str):
        """
        Deserialize the index from disk.
        :param path: path to the index
        """
        logger.info("Loading index from %s", path)
        index_file, meta_file = self.get_files(path)

        self.index = faiss.read_index(index_file)
        logger.info("Loaded index of type %s and size %d", type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def _update_id_mapping(self, db_ids: List) -> int:
        self.index_id_to_db_id.extend(db_ids)
        return len(self.index_id_to_db_id)


class DenseFlatIndexer(DenseIndexer):
    def __init__(self, buffer_size: int = 50000):
        """
        Initialize a DenseFlatIndexer
        :param buffer_size: size of the buffer to use when indexing data
        """
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)

    def init_index(self, vector_sz: int):
        """
        Initialize the index.
        :param vector_sz: size of the vectors to index
        """
        self.index = faiss.index_factory(vector_sz, "IVF64,Flat", faiss.METRIC_INNER_PRODUCT)
        
        res = faiss.StandardGpuResources() # use a single GPU
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)


    def index_data(self, data: List[Tuple[object, np.array]]):
        """
        Index the given data.
        :param data: list of tuples of (id, vector)
        """
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            db_ids = [t[0] for t in data[i : i + self.buffer_size]]
            vectors = [np.reshape(t[1], (1, -1)) for t in data[i : i + self.buffer_size]]
            vectors = np.concatenate(vectors, axis=0)
            total_data = self._update_id_mapping(db_ids)
            self.index.add(vectors)
            logger.info("data indexed %d", total_data)

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info("Total data indexed %d", indexed_cnt)

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        """
        Search for the top_docs nearest neighbors of the given query vectors.
        :param query_vectors: query vectors
        :param top_docs: number of nearest neighbors to return
        :return: list of tuples of (id, distance)
        """
        scores, indexes = self.index.search(query_vectors, top_docs)

        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        return db_ids, scores

    def get_index_name(self):
        return "flat_index"


class DenseHNSWFlatIndexer(DenseIndexer):
    """
    Efficient index for retrieval. Note: default settings are for high accuracy but also high RAM usage
    """

    def __init__(
        self,
        buffer_size: int = 1e9,
        store_n: int = 512,
        ef_search: int = 128,
        ef_construction: int = 200,
    ):
        """
        Initialize a DenseHNSWFlatIndexer
        :param buffer_size: size of the buffer to use when indexing data
        :param store_n: number of vectors to store in the index
        :param ef_search: expected number of neighbors to search for
        :param ef_construction: expected number of neighbors to construct the index
        """
        super(DenseHNSWFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.store_n = store_n
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.phi = 0

    def init_index(self, vector_sz: int):
        """
        Initialize the index.
        :param vector_sz: size of the vectors to index
        """
        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWFlat(vector_sz + 1, self.store_n)
        index.hnsw.efSearch = self.ef_search
        index.hnsw.efConstruction = self.ef_construction
        self.index = index

    def index_data(self, data: List[Tuple[object, np.array]]):
        """
        Index the given data.
        :param data: list of tuples of (id, vector)
        """
        n = len(data)

        # max norm is required before putting all vectors in the index to convert inner product similarity to L2
        if self.phi > 0:
            raise RuntimeError(
                "DPR HNSWF index needs to index all data at once," "results will be unpredictable otherwise."
            )
        phi = 0
        for i, item in enumerate(data):
            id, doc_vector = item[0:2]
            norms = (doc_vector ** 2).sum()
            phi = max(phi, norms)
        logger.info("HNSWF DotProduct -> L2 space phi={}".format(phi))
        self.phi = phi

        # indexing in batches is beneficial for many faiss index types
        bs = int(self.buffer_size)
        for i in range(0, n, bs):
            db_ids = [t[0] for t in data[i : i + bs]]
            vectors = [np.reshape(t[1], (1, -1)) for t in data[i : i + bs]]

            norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
            aux_dims = [np.sqrt(phi - norm) for norm in norms]
            hnsw_vectors = [np.hstack((doc_vector, aux_dims[i].reshape(-1, 1))) for i, doc_vector in enumerate(vectors)]
            hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)
            self.train(hnsw_vectors)

            self._update_id_mapping(db_ids)
            self.index.add(hnsw_vectors)
            logger.info("data indexed %d", len(self.index_id_to_db_id))
        indexed_cnt = len(self.index_id_to_db_id)
        logger.info("Total data indexed %d", indexed_cnt)

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        """
        Search for the top_docs nearest neighbors of the given query vectors.
        :param query_vectors: query vectors
        :param top_docs: number of nearest neighbors to return
        :return: list of tuples of (id, distance)
        """
        aux_dim = np.zeros(len(query_vectors), dtype="float32")
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        logger.info("query_hnsw_vectors %s", query_nhsw_vectors.shape)
        scores, indexes = self.index.search(query_nhsw_vectors, top_docs)

        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        return db_ids, scores

    def deserialize(self, file: str):
        """
        Deserialize the index from the given file.
        :param file: file to deserialize the index from
        """
        super(DenseHNSWFlatIndexer, self).deserialize(file)
        # to trigger exception on subsequent indexing
        self.phi = 1

    def get_index_name(self):
        return "hnsw_index"


class DenseHNSWSQIndexer(DenseHNSWFlatIndexer):
    """
    Efficient index for retrieval. Note: default settings are for high accuracy but also high RAM usage
    """

    def __init__(
        self,
        buffer_size: int = 1e10,
        store_n: int = 128,
        ef_search: int = 128,
        ef_construction: int = 200,
    ):
        """
        Initialize a DenseHNSWSQIndexer
        :param buffer_size: size of the buffer to use when indexing data
        :param store_n: number of vectors to store in the index
        :param ef_search: expected number of neighbors to search for
        :param ef_construction: expected number of neighbors to construct the index
        """
        super(DenseHNSWSQIndexer, self).__init__(
            buffer_size=buffer_size,
            store_n=store_n,
            ef_search=ef_search,
            ef_construction=ef_construction,
        )

    def init_index(self, vector_sz: int):
        """
        Initialize the index.
        :param vector_sz: size of the vectors to index
        """
        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWSQ(vector_sz + 1, faiss.ScalarQuantizer.QT_8bit, self.store_n)
        index.hnsw.efSearch = self.ef_search
        index.hnsw.efConstruction = self.ef_construction
        self.index = index

    def train(self, vectors: np.array):
        """
        Train the index.
        :param vectors: vectors to train the index with
        """
        self.index.train(vectors)

    def get_index_name(self):
        """
        Get the name of the index.
        """
        return "hnswsq_index"