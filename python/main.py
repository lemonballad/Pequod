import matplotlib.pyplot as plt  # Data visualization and plotting
import networkx as nx
import sympy as sp  # Symbolic mathematics for manipulating expressions
from utils import ExprGraphProcessor, graph_to_expr

# Example usage
if __name__ == "__main__":
    x, y, z = sp.symbols('x y z')
    expr = sp.Eq(sp.Derivative(y * x, x, y) + x, y + z + sp.sin(x))

    # Initialize the graph processor and process the expression
    graph = nx.DiGraph()
    processor = ExprGraphProcessor()
    _, graph = processor.process_expr(expr, graph, side_flag=0)

    # Plot the graph
    pos = nx.circular_layout(graph)
    labels = nx.get_node_attributes(graph, 'label')
    nx.draw(graph, pos=pos, with_labels=True, labels=labels, arrows=True, font_size=10, font_color='black')
    plt.show()

    root_node = 0  # Adjust this according to the actual root node of your graph
    reconstructed_expr = converter.graph_to_expr(graph, root_node)
    print(reconstructed_expr)
