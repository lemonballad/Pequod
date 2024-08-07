import unittest
import sympy as sp
import networkx as nx
import logging
import matplotlib.pyplot as plt
from python.utils import ExprGraphProcessor, ExprGenerator, GraphToExprConverter, radial_layout, draw_graph

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TestExprGraphProcessor(unittest.TestCase):

    def setUp(self):
        logging.info("Setting up ExprGraphProcessor test case")
        self.processor = ExprGraphProcessor()
        self.graph = nx.DiGraph()

    def test_reset_counters(self):
        logging.info("Testing reset_counters method")
        self.processor.node_count_neg = -5
        self.processor.node_count_pos = 5
        self.processor.reset_counters()
        self.assertEqual(self.processor.node_count_neg, 0)
        self.assertEqual(self.processor.node_count_pos, 0)

    def test_get_next_node(self):
        logging.info("Testing get_next_node method")
        self.processor.reset_counters()
        self.processor.set_imaginary_part(0)
        self.assertEqual(self.processor.get_next_node(0), 0)
        self.assertEqual(self.processor.get_next_node(-1), -1)
        self.assertEqual(self.processor.get_next_node(1), 1)
        self.assertEqual(self.processor.get_next_node(-1), -2)
        self.assertEqual(self.processor.get_next_node(1), 2)

    def test_add_node(self):
        logging.info("Testing add_node method")
        node_name = self.processor.add_node(self.graph, 1, 'test_node')
        self.assertTrue(node_name in self.graph)
        self.assertEqual(self.graph.nodes[node_name]['label'], 'test_node')

    def test_handle_symbol(self):
        logging.info("Testing handle_symbol method")
        x = sp.Symbol('x')
        node, graph = self.processor.handle_symbol(x, self.graph, 1)
        self.assertTrue(node in graph)
        self.assertEqual(graph.nodes[node]['label'], 'x')
        self.assertEqual(graph.nodes[node]['variable'], x)

    def test_process_expr(self):
        logging.info("Testing process_expr method")
        expr = sp.Symbol('w') + sp.Symbol('x')
        self.processor.set_imaginary_part(0)
        node, graph = self.processor.process_expr(expr, self.graph, 1)
        self.assertTrue(node in graph)
        self.assertEqual(graph.nodes[node]['label'], '+')

    def test_process_iterable(self):
        logging.info("Testing process_iterable method")
        x, y, z = sp.symbols('x y z')
        expr1 = sp.Eq(sp.Derivative(y * x, x, y) + x, y + z + sp.sin(x) - sp.Function('F')(x))
        expr2 = sp.Eq(x ** 2 + y ** 2, z ** 2)
        graph = self.processor.process_iterable([expr1, expr2])
        self.assertTrue(len(graph) > 0)
        self.assertTrue(any(data.get('operator') == '=' for _, data in graph.nodes(data=True)))


class TestExprGenerator(unittest.TestCase):

    def setUp(self):
        logging.info("Setting up ExprGenerator test case")
        self.generator = ExprGenerator([sp.Symbol('x'), sp.Symbol('y')])

    def test_generate_polynomials(self):
        logging.info("Testing generate_polynomials method")
        polynomials = self.generator.get_polynomials()
        self.assertTrue(len(polynomials) > 0)
        self.assertTrue(any(isinstance(expr, sp.Add) for expr in polynomials))

    def test_generate_trigonometric(self):
        logging.info("Testing generate_trigonometric method")
        trig_expressions = self.generator.get_trigonometric()
        self.assertTrue(len(trig_expressions) > 0)
        self.assertTrue(any(isinstance(expr, sp.sin) for expr in trig_expressions))

    def test_generate_hyperbolic(self):
        logging.info("Testing generate_hyperbolic method")
        hyperbolic_expressions = self.generator.get_hyperbolic()
        self.assertTrue(len(hyperbolic_expressions) > 0)
        self.assertTrue(any(isinstance(expr, sp.sinh) for expr in hyperbolic_expressions))

    def test_generate_exponential(self):
        logging.info("Testing generate_exponential method")
        exponential_expressions = self.generator.get_exponential()
        self.assertTrue(len(exponential_expressions) > 0)
        self.assertTrue(any(isinstance(expr, sp.exp) for expr in exponential_expressions))


class TestGraphToExprConverter(unittest.TestCase):

    def setUp(self):
        logging.info("Setting up GraphToExprConverter test case")
        self.converter = GraphToExprConverter()

    def test_convert_graph(self):
        logging.info("Testing convert_graph method")
        x, y, z = sp.symbols('x y z')
        expr1 = sp.Eq(sp.Derivative(y * x, x, y) + x, y + z + sp.sin(x) - sp.Function('F')(x))
        expr2 = sp.Eq(x ** 2 + y ** 2, z ** 2)
        processor = ExprGraphProcessor()
        graph = processor.process_iterable([expr1, expr2])
        expressions = self.converter.convert_graph(graph)
        self.assertEqual(len(expressions), 2)
        for expr in expressions:
            self.assertTrue(isinstance(expr, sp.Equality))

    def test_conversion_accuracy(self):
        logging.info("Testing conversion accuracy")
        x, y, z = sp.symbols('x y z')
        expr1 = sp.Eq(sp.Derivative(y * x, x, y) + x, y + z + sp.sin(x) - sp.Function('F')(x))
        processor = ExprGraphProcessor()
        graph = processor.process_iterable([expr1])
        expressions = self.converter.convert_graph(graph)
        self.assertEqual(len(expressions), 1)
        self.assertTrue(sp.simplify(expr1.lhs - expressions[0].lhs) == 0)
        self.assertTrue(sp.simplify(expr1.rhs - expressions[0].rhs) == 0)


class TestGraphVisualization(unittest.TestCase):

    def setUp(self):
        logging.info("Setting up GraphVisualization test case")
        self.processor = ExprGraphProcessor()
        self.converter = GraphToExprConverter()

    def test_radial_layout(self):
        logging.info("Testing radial_layout function")
        x, y, z = sp.symbols('x y z')
        expr = sp.Eq(x + y, z)
        graph = self.processor.process_expr(expr, nx.DiGraph(), 1)[1]
        pos = radial_layout(graph)
        self.assertTrue(isinstance(pos, dict))
        self.assertTrue(len(pos) > 0)

    def test_draw_graph(self):
        logging.info("Testing draw_graph function")
        x, y, z = sp.symbols('x y z')
        expr = sp.Eq(x + y, z)
        graph = self.processor.process_expr(expr, nx.DiGraph(), 1)[1]
        pos = radial_layout(graph)
        labels = nx.get_node_attributes(graph, 'label')
        draw_graph(graph, pos, labels)
        plt.close()  # Close the plot after drawing to avoid displaying during tests


if __name__ == '__main__':
    unittest.main()
