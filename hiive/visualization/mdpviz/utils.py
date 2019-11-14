import os

import networkx as nx

from hiive.visualization.mdpviz import MDPSpec


def graph_to_png(graph):
    pydot_graph = nx.nx_pydot.to_pydot(graph)
    return pydot_graph.create_png()


def write_to_png(graph, file, dpi=300, **kwargs):
    try:
        os.makedirs(os.path.abspath(os.path.dirname(file)))
    except:
        pass
    pydot_graph = nx.nx_pydot.to_pydot(graph)
    pydot_graph.set('dpi', dpi)
    for k in kwargs:
        pydot_graph.set(k, kwargs[k])
    pydot_graph.write_png(file)


def display_mdp(mdp: MDPSpec):
    from IPython.display import display, Image
    display(Image(graph_to_png(mdp.to_graph())))