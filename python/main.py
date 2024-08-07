import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sympy as sp
from utils import ExprGraphProcessor, GraphToExprConverter


def radial_layout(_graph_, _root_node_=0):
    """
    Creates a radial layout for the graph with the root node in the center.

    Args:
        _graph_ (nx.DiGraph): The graph to layout.
        _root_node_ (complex): The root node which should be at the center.

    Returns:
        dict: A dictionary with node positions.
    """
    _pos_ = {}
    _pos_[0] = [0, 0]  # Place root node at the center
    equals_nodes = [node for node, data in _graph_.nodes(data=True) if data.get('operator') == '=']

    if len(equals_nodes) == 1:
        _pos_[equals_nodes[0]] = [1, 0]
        equation_nodes = [node for node, data in _graph_.nodes(data=True) if np.imag(node) == 1 and np.real(node) != 0]
        _pos_.update(nx.circular_layout(equation_nodes, center=_pos_[equals_nodes[0]], scale=0.75))
    else:
        _pos_.update(nx.circular_layout(equals_nodes, center=_pos_[0]))
        for equals_node in equals_nodes:
            equation_nodes = [node for node, data in _graph_.nodes(data=True) if np.imag(node) == np.imag(equals_node)
                              and np.real(node) != 0]
            _pos_.update(nx.circular_layout(equation_nodes, center=_pos_[equals_node], scale=0.75))
    return _pos_


def draw_graph(_graph_, _pos_, _labels_):
    """
    Draw the graph with a specified color scheme and optional black background.

    Args:
        _graph_ (nx.DiGraph): The graph to draw.
        _pos_ (dict): Node positions.
        _labels_ (dict): Node labels.
    """
    # Color Universal Design (CUD) palette
    cud_colors = {
        'default': '#E69F00',  # Orange
        'exists': '#56B4E9',  # Sky Blue
        'symbol': '#009E73',  # Bluish Green
        'number': '#F0E442',  # Yellow
        'operator': '#0072B2',  # Blue
        'function': '#D55E00',  # Vermilion
        '=': '#CC79A7'  # Reddish Purple
    }

    node_colors = []
    for node, data in _graph_.nodes(data=True):
        if data.get('label') == r'$\exists$':
            node_colors.append(cud_colors['exists'])
        elif data.get('operator') == '=':
            node_colors.append(cud_colors['='])
        elif data.get('operator') == 'symbol':
            node_colors.append(cud_colors['symbol'])
        elif data.get('operator') == 'number':
            node_colors.append(cud_colors['number'])
        elif data.get('operator') in ['+', '*', '**']:
            node_colors.append(cud_colors['operator'])
        elif data.get('operator') == 'function':
            node_colors.append(cud_colors['function'])
        else:
            node_colors.append(cud_colors['default'])

    plt.figure(figsize=(12, 12))

    nx.draw(_graph_, pos=_pos_, with_labels=True, labels=_labels_, arrows=True, font_size=10,
            node_size=2000, node_color=node_colors)
    plt.show()


# Example usage
if __name__ == "__main__":
    x, y, z = sp.symbols('x y z')
    F = sp.Function('F')
    expr1 = sp.Eq(sp.Derivative(y * x, x, y) + x, y + z + sp.sin(x) - F(x))
    expr2 = sp.Eq(x ** 2 + y ** 2, z ** 2)

    processor = ExprGraphProcessor()
    graph = processor.process_iterable([expr1, expr2])  # Use a list to pass the expression as an iterable

    pos = radial_layout(graph)
    labels = nx.get_node_attributes(graph, 'label')

    draw_graph(graph, pos, labels)

    converter = GraphToExprConverter()
    reconstructed_expressions = converter.convert_graph(graph)
    for original_expr, reconstructed_expr in zip([expr1, expr2], reconstructed_expressions):
        print("Original Expression:", original_expr)
        print("Reconstructed Expression:", reconstructed_expr)
        print("Difference:",
              sp.Eq(original_expr.lhs - reconstructed_expr.lhs, original_expr.rhs - reconstructed_expr.rhs))
