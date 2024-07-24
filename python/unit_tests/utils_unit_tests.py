import unittest
import sympy as sp
import networkx as nx
from python.utils import ExprGraphProcessor, ExprGenerator


class TestExprGraphProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = ExprGraphProcessor()
        self.graph = nx.DiGraph()

    def test_reset_counters(self):
        self.processor.node_count_neg = -5
        self.processor.node_count_pos = 5
        self.processor.reset_counters()
        self.assertEqual(self.processor.node_count_neg, 0)
        self.assertEqual(self.processor.node_count_pos, 0)

    def test_get_next_node(self):
        self.processor.reset_counters()
        self.assertEqual(self.processor.get_next_node(0), 0)
        self.assertEqual(self.processor.get_next_node(-1), -1)
        self.assertEqual(self.processor.get_next_node(1), 1)
        self.assertEqual(self.processor.get_next_node(-1), -2)
        self.assertEqual(self.processor.get_next_node(1), 2)

    def test_add_node(self):
        node_name = self.processor.add_node(self.graph, 1, 'test_node')
        self.assertTrue(node_name in self.graph)
        self.assertEqual(self.graph.nodes[node_name]['label'], '1: test_node')

    def test_handle_symbol(self):
        x = sp.Symbol('x')
        node, graph = self.processor.handle_symbol(x, self.graph, 1)
        self.assertTrue(node in graph)
        self.assertEqual(graph.nodes[node]['label'], '1: x')
        self.assertEqual(graph.nodes[node]['symbol'], x)

    def test_process_expr(self):
        expr = sp.Symbol('x') + sp.Symbol('y')
        node, graph = self.processor.process_expr(expr, self.graph, 1)
        self.assertTrue(node in graph)
        self.assertEqual(graph.nodes[node]['label'], '1: +')


class TestExprGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = ExprGenerator([sp.Symbol('x'), sp.Symbol('y')])

    def test_generate_polynomials(self):
        polynomials = self.generator.get_polynomials()
        self.assertTrue(len(polynomials) > 0)
        self.assertTrue(any(isinstance(expr, sp.Add) for expr in polynomials))

    def test_generate_trigonometric(self):
        trig_expressions = self.generator.get_trigonometric()
        self.assertTrue(len(trig_expressions) > 0)
        self.assertTrue(any(isinstance(expr, sp.sin) for expr in trig_expressions))

    def test_generate_hyperbolic(self):
        hyperbolic_expressions = self.generator.get_hyperbolic()
        self.assertTrue(len(hyperbolic_expressions) > 0)
        self.assertTrue(any(isinstance(expr, sp.sinh) for expr in hyperbolic_expressions))

    def test_generate_exponential(self):
        exponential_expressions = self.generator.get_exponential()
        self.assertTrue(len(exponential_expressions) > 0)
        self.assertTrue(any(isinstance(expr, sp.exp) for expr in exponential_expressions))


if __name__ == '__main__':
    unittest.main()
