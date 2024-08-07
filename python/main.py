import matplotlib.pyplot as plt
import networkx as nx
import sympy as sp
import logging
from python.utils import ExprGraphProcessor, GraphToExprConverter, radial_layout, draw_graph

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    logging.info("Starting the main function")

    x, y, z = sp.symbols('x y z')
    F = sp.Function('F')
    expr1 = sp.Eq(sp.Derivative(y * x, x, y) + x, y + z + sp.sin(x) - F(x))
    expr2 = sp.Eq(x ** 2 + y ** 2, z ** 2)

    # Initialize the graph processor and process the expression
    processor = ExprGraphProcessor()
    logging.info("Processing the expressions into a graph")
    graph = processor.process_iterable([expr1, expr2])

    # Plot the graph
    pos = radial_layout(graph)
    labels = nx.get_node_attributes(graph, 'label')
    draw_graph(graph, pos, labels)
    plt.show()

    # Example conversion and comparison
    converter = GraphToExprConverter()
    logging.info("Converting the graph back to expressions")
    reconstructed_expressions = converter.convert_graph(graph)
    for original_expr, reconstructed_expr in zip([expr1, expr2], reconstructed_expressions):
        logging.info(f"Original Expression: {original_expr}")
        logging.info(f"Reconstructed Expression: {reconstructed_expr}")
        logging.info(
            f"Difference: {sp.Eq(original_expr.lhs - reconstructed_expr.lhs, original_expr.rhs - reconstructed_expr.rhs)}")


if __name__ == "__main__":
    main()
