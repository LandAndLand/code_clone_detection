from typing import Dict
from queue import Queue

import networkx as nx
import numpy as np
from gensim.models.keyedvectors import Word2VecKeyedVectors
from matplotlib import pyplot as plt
from tree_sitter import Language, Parser, Node

import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

JAVA_SO_PATH: str = "../data/java.so"


class TreeSitterNode(object):

    def __init__(self, node: Node, program: str = None):
        """
        :param node: The tree_sitter node
        :param program: the str of the program
        """

        self.type = node.type
        self.start_byte = node.start_byte
        self.end_byte = node.end_byte
        self.name = self.get_name(node, program)

    def get_name(self, node: Node, program: str = None) -> str:
        if program is None:
            return node.type

        if 'identifier' in node.type and node.is_named:
            return program[self.start_byte:self.end_byte]
        else:
            return node.type

    def __eq__(self, obj):
        return self.type == obj.type and self.start_byte == obj.start_byte and self.end_byte == obj.end_byte

    def __str__(self):
        return f'{self.name} @ [{self.start_byte}, {self.end_byte}]'

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())


def get_parser(so_path: str = None) -> Parser:
    if so_path is None:
        so_path = JAVA_SO_PATH

    JAVA_LANGUAGE = Language(so_path, 'java')

    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)

    return parser


def parse_program(program: str, parser: Parser = None, code2vec: Word2VecKeyedVectors = None) -> nx.DiGraph:
    if parser is None:
        parser: Parser = get_parser()

    tree = parser.parse(bytes(program, "utf8"))

    # 建立一个空的有向图
    g: nx.DiGraph = nx.DiGraph()

    queue: Queue = Queue()
    queue.put(tree.root_node)

    while not queue.empty():
        # 按照宽度优先的顺序来建立一个有向图
        node = queue.get()

        if not hasattr(node, 'children'):
            continue
        
        # 依次将父节点与子节点连接起来：root-child 建立边的关系
        for child in node.children:
            g.add_edge(TreeSitterNode(node, program), TreeSitterNode(child, program))

            queue.put(child)

    # embedding are added to each node
    # 使用code2vec的嵌入表示来初始化表示图中的节点
    if code2vec is not None:
        zeros = np.zeros(code2vec.vector_size)
        for node in g.nodes:
            name = node.name.lower()
            if name in code2vec:
                g.add_node(node, data=code2vec.get_vector(name))
            else:
                g.add_node(node, data=zeros)

    return g


def plot_graph(g: nx.DiGraph):
    from networkx.drawing.nx_agraph import graphviz_layout

    pos = graphviz_layout(g, prog='dot')
    fig, ax = plt.subplots(1, 1, figsize=(40, 20))
    labels: Dict[TreeSitterNode, str] = {node: node.name for node in g.nodes}
    nx.draw(g, pos, ax=ax, with_labels=True, labels=labels, arrows=True, font_size=15, node_color="yellow")


def get_data_from_graph(g: nx.DiGraph, y=None) -> Data:
    # 返回图邻接矩阵。graph_spicy.todense()是一个二维矩阵，矩阵的每一个元素是两个节点之间的边的权重
    graph_spicy = nx.to_scipy_sparse_matrix(g, format='coo')
    edge_index = torch.tensor([graph_spicy.row, graph_spicy.col], dtype=torch.long)

    if len(nx.get_node_attributes(g, "data")) > 0:
        x = torch.tensor([x[1] for x in g.nodes(data='data')], dtype=torch.float)
    else:
        x = torch.tensor(graph_spicy.data, dtype=torch.float)

    if not y is None:
        y = int(y)
        y = torch.tensor([y], dtype=torch.int64)

    data = Data(x=x, edge_index=edge_index, y=y)

    return data


def program_to_data(program: str, parser: Parser = None) -> Data:
    return get_data_from_graph(parse_program(program, parser))


if __name__ == '__main__':
    program_str = """
        public int add(int a, int b) {
            int c = 0;
            c = a + b;
            return c;
        }
    """

    program_str_2 = """
        public int add_numbers(int c, int d) {
            int e = 0;
            return c + d;
        }
    """

    parser = get_parser(JAVA_SO_PATH)
    g1 = parse_program(program_str, parser)
    g2 = parse_program(program_str_2, parser)
    g3 = nx.compose(g1, g2)
    plot_graph(g3)
    plt.show()
