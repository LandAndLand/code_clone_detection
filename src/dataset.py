import glob
import os
import pickle
import random

from torch_geometric.data import Dataset

from code_parser import *
from code_parser import  get_data_from_graph

# train_dataset = FastCloneDataset(root="../data/", functions_path="../data/networkx_data_idx/", pairs_path="../data/train.npz", return_pair_data=False)
class FastCloneDataset(Dataset):
    '''
    FastCloneDataset： 

    networkx_data_idx 存储的是: source code的语法树形式，且每个语法树的node已经被fasttext嵌入表示
    '''
    def __init__(self, root, functions_path, pairs_path, return_pair_data=False, transform=None, pre_transform=None):
        self.functions_path = functions_path
        self.return_pair_data = return_pair_data
        # 训练集或测试集中存储的pair
        self.pairs = np.load(pairs_path)['arr_0']
        self.functions = {}
        self.processed_pairs = {}
        super(FastCloneDataset, self).__init__(root, transform, pre_transform)

    def len(self):
        return len(self.pairs)

    def load_id(self, idx):
        if idx in self.functions:
            return self.functions[idx]

        g = nx.read_gpickle(os.path.join(self.functions_path, str(idx)))

        if self.return_pair_data:
            self.functions[idx] = get_data_from_graph(g)
        else:
            self.functions[idx] = g

        return self.functions[idx]

    def get_pair(self, idx):
        if idx in self.processed_pairs:
            return self.processed_pairs[idx]

        id1, id2, label = self.pairs[idx]
        data1, data2 = self.load_id(id1), self.load_id(id2)
        if self.return_pair_data:
            data = PairData(edge_index_s=data1.edge_index, x_s=data1.x, edge_index_t=data2.edge_index, x_t=data2.x,
                            y=torch.tensor([label], dtype=torch.int64))
            self.processed_pairs[idx] = data
        else:
            g1, g2 = data1, data2
            g3 = nx.disjoint_union(g1, g2)
            # for node1 in g1.nodes(data=True):
            #     for node2 in g2.nodes(data=True):
            #         if node1[1]['idx'] == node2[1]['idx']:
            #             g3.add_edge("s_" + str(node1[0]), "t_" + str(node2[0]))

            g3 = nx.convert_node_labels_to_integers(g3.to_undirected())
            self.processed_pairs[idx] = get_data_from_graph(g3, label)

        return self.processed_pairs[idx]

    def get(self, idx):
        return self.get_pair(idx)


class CloneDataset(Dataset):
    def __init__(self, root, functions_path, pairs_path, transform=None, pre_transform=None):

        self.functions_path = functions_path
        self.pairs_path = pairs_path

        self.pair_ids = np.load(pairs_path)['arr_0']

        super(CloneDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.functions_path)

    @property
    def processed_file_names(self):

        processed_file_names = []

        for idx in range(len(self.pair_ids)):
            path = os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))
            if os.path.exists(path):
                processed_file_names.append('data_{}.pt'.format(idx))

        if len(processed_file_names) < 10:
            return ['data_{}.pt'.format(idx) for idx in range(len(self.pair_ids))]

        return processed_file_names

    def process(self):

        global apply

        def apply(in_):
            i, (id1, id2, label) = in_
            try:
                with open(os.path.join(self.functions_path, str(id1)), 'rb') as f:
                    g1: nx.DiGraph = pickle.load(f)
                with open(os.path.join(self.functions_path, str(id2)), 'rb') as f:
                    g2: nx.DiGraph = pickle.load(f)
            except FileNotFoundError as ex:
                print(ex)
                return

            g3 = nx.compose(g1, g2)

            y = int(label)

            data: Data = get_data_from_graph(g3, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                return

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))

        from multiprocessing import Pool
        with Pool(40) as p:
            p.map(apply, enumerate(self.pair_ids))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        path = os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))

        if not os.path.exists(path):
            path = random.choice(glob.glob(os.path.join(self.processed_dir, "data_*.pt")))
            print("random", end="\r")

        data = torch.load(path)
        return data


class PairData(Data):
    def __init__(self, edge_index_s, x_s, edge_index_t, x_t, y):

        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.y = y
        super(PairData, self).__init__(y=y)

    def __inc__(self, key, value):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super(PairData, self).__inc__(key, value)


class CloneDatasetPair(Dataset):
    def __init__(self, root, functions_path, pairs_path, transform=None, pre_transform=None):

        self.functions_path = functions_path
        self.pairs_path = pairs_path

        self.pair_ids = np.load(pairs_path)['arr_0']

        super(CloneDatasetPair, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.functions_path)

    @property
    def processed_file_names(self):

        processed_file_names = []

        for idx in range(len(self.pair_ids)):
            path = os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))
            if os.path.exists(path):
                processed_file_names.append('data_{}.pt'.format(idx))

        if len(processed_file_names) < 10:
            return ['data_{}.pt'.format(idx) for idx in range(len(self.pair_ids))]

        return processed_file_names

    def process(self):

        global apply

        def apply(in_):
            i, (id1, id2, label) = in_
            try:
                with open(os.path.join(self.functions_path, str(id1)), 'rb') as f:
                    g1: nx.DiGraph = pickle.load(f)
                with open(os.path.join(self.functions_path, str(id2)), 'rb') as f:
                    g2: nx.DiGraph = pickle.load(f)
            except FileNotFoundError as ex:
                print(ex)
                return

            data1: Data = get_data_from_graph(g1, y=label)
            data2: Data = get_data_from_graph(g2, y=label)
            data = PairData(edge_index_s=data1.edge_index, x_s=data1.x, edge_index_t=data2.edge_index, x_t=data2.x,
                            y=data2.y)

            if self.pre_filter is not None and not self.pre_filter(data):
                return

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))

        from multiprocessing import Pool
        with Pool(40) as p:
            p.map(apply, enumerate(self.pair_ids))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        path = os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))

        if not os.path.exists(path):
            path = random.choice(glob.glob(os.path.join(self.processed_dir, "data_*.pt")))
            print("random")

        data = torch.load(path)
        return data
