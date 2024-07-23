import sympy as sp  # Symbolic mathematics for manipulating expressions
import networkx as nx
from typing import Tuple, Union


class ExprGraphProcessor:
    """
    A class to process sympy expressions and convert them into a NetworkX directed graph.
    """

    def __init__(self):
        """
        Initializes the processor and resets node counters.
        """
        self.node_count_neg = None
        self.node_count_pos = None
        self.reset_counters()

    def reset_counters(self):
        """
        Resets the counters for node naming based on the side of the expression.
        """
        self.node_count_pos = 0  # Counter for positive node names
        self.node_count_neg = 0  # Counter for negative node names

    def get_next_node(self, side_flag: int) -> int:
        """
        Determines the next node name based on the side of the expression.

        Args:
            side_flag (int): Flag indicating side of the equation (-1: lhs, 1: rhs, 0: initial call).

        Returns:
            int: The next node name.
        """
        if side_flag == 0:
            return 0
        elif side_flag < 0:
            self.node_count_neg -= 1
            return self.node_count_neg
        else:
            self.node_count_pos += 1
            return self.node_count_pos

    @staticmethod
    def add_edge(graph: nx.DiGraph, from_node: int, to_node: int, **attrs):
        """
        Adds an edge to the graph.

        Args:
            graph (nx.DiGraph): The graph to which the edge will be added.
            from_node (int): The start node of the edge.
            to_node (int): The end node of the edge.
            **attrs: Additional attributes for the edge.
        """
        graph.add_edge(from_node, to_node, **attrs)

    def add_node(self, _graph_: nx.DiGraph, side_flag: int, label: str, **attrs) -> int:
        """
        Adds a node to the graph with a unique name and label.

        Args:
            _graph_ (nx.DiGraph): The graph to which the node will be added.
            side_flag (int): Flag indicating side of the equation (-1: lhs, 1: rhs, 0: initial call).
            label (str): The label for the node.
            **attrs: Additional attributes for the node.

        Returns:
            int: The name of the newly added node.
        """
        node_name = self.get_next_node(side_flag)
        _graph_.add_node(node_name, label=f'{node_name}: {label}', **attrs)
        return node_name

    def handle_equality(self, _expr_: sp.Equality, _graph_: nx.DiGraph, side_flag: int) -> Tuple[int, nx.DiGraph]:
        """
        Handles equality expressions and adds them to the graph.

        Args:
            _expr_ (sp.Equality): The equality expression.
            _graph_ (nx.DiGraph): The graph to which the nodes and edges will be added.
            side_flag (int): Flag indicating side of the equation (-1: lhs, 1: rhs).

        Returns:
            Tuple[int, nx.DiGraph]: The node representing the equality and the updated graph.
        """
        eq_node = self.add_node(_graph_, side_flag=0, label='=', operator='=')
        lhs_node, _graph_ = self.process_expr(_expr_.lhs, _graph_, side_flag=-1)
        rhs_node, _graph_ = self.process_expr(_expr_.rhs, _graph_, side_flag=1)
        self.add_edge(_graph_, lhs_node, eq_node)
        self.add_edge(_graph_, rhs_node, eq_node)
        return eq_node, _graph_

    def handle_add(self, _expr_: sp.Add, _graph_: nx.DiGraph, side_flag: int) -> Tuple[int, nx.DiGraph]:
        """
        Handles addition expressions and adds them to the graph.

        Args:
            _expr_ (sp.Add): The addition expression.
            _graph_ (nx.DiGraph): The graph to which the nodes and edges will be added.
            side_flag (int): Flag indicating side of the equation.

        Returns:
            Tuple[int, nx.DiGraph]: The node representing the addition and the updated graph.
        """
        add_node = self.add_node(_graph_, side_flag, label='+', operator='+')
        for term in _expr_.args:
            term_node, _graph_ = self.process_expr(term, _graph_, side_flag)
            self.add_edge(_graph_, term_node, add_node)
        return add_node, _graph_

    def handle_mul(self, _expr_: sp.Mul, _graph_: nx.DiGraph, side_flag: int) -> Tuple[int, nx.DiGraph]:
        """
        Handles multiplication expressions and adds them to the graph.

        Args:
            _expr_ (sp.Mul): The multiplication expression.
            _graph_ (nx.DiGraph): The graph to which the nodes and edges will be added.
            side_flag (int): Flag indicating side of the equation.

        Returns:
            Tuple[int, nx.DiGraph]: The node representing the multiplication and the updated graph.
        """
        mul_node = self.add_node(_graph_, side_flag, label='*', operator='*')
        for term in _expr_.args:
            term_node, _graph_ = self.process_expr(term, _graph_, side_flag)
            self.add_edge(_graph_, term_node, mul_node)
        return mul_node, _graph_

    def handle_pow(self, _expr_: sp.Pow, _graph_: nx.DiGraph, side_flag: int) -> Tuple[int, nx.DiGraph]:
        """
        Handles power expressions and adds them to the graph.

        Args:
            _expr_ (sp.Pow): The power expression.
            _graph_ (nx.DiGraph): The graph to which the nodes and edges will be added.
            side_flag (int): Flag indicating side of the equation.

        Returns:
            Tuple[int, nx.DiGraph]: The node representing the power operation and the updated graph.
        """
        pow_node = self.add_node(_graph_, side_flag, label='**', operator='**')
        base_node, _graph_ = self.process_expr(_expr_.args[0], _graph_, side_flag)
        self.add_edge(_graph_, pow_node, base_node)
        exponent_node, _graph_ = self.process_expr(_expr_.args[1], _graph_, side_flag)
        self.add_edge(_graph_, exponent_node, pow_node)
        return pow_node, _graph_

    def handle_symbol(self, _expr_: sp.Symbol, _graph_: nx.DiGraph, side_flag: int) -> Tuple[int, nx.DiGraph]:
        """
        Handles symbol expressions and adds them to the graph.

        Args:
            _expr_ (sp.Symbol): The symbol expression.
            _graph_ (nx.DiGraph): The graph to which the nodes will be added.
            side_flag (int): Flag indicating side of the equation.

        Returns:
            Tuple[int, nx.DiGraph]: The node representing the symbol and the updated graph.
        """
        symbol_node = self.add_node(_graph_, side_flag, label=str(_expr_), symbol=_expr_)
        return symbol_node, _graph_

    def handle_number(self, _expr_: Union[sp.Integer, sp.Float], _graph_: nx.DiGraph, side_flag: int) -> Tuple[
        int, nx.DiGraph]:
        """
        Handles number expressions and adds them to the graph.

        Args:
            _expr_ (Union[sp.Integer, sp.Float]): The number expression.
            _graph_ (nx.DiGraph): The graph to which the nodes will be added.
            side_flag (int): Flag indicating side of the equation.

        Returns:
            Tuple[int, nx.DiGraph]: The node representing the number and the updated graph.
        """
        number_node = self.add_node(_graph_, side_flag, label=str(_expr_), number=_expr_)
        return number_node, _graph_

    def handle_derivative(self, _expr_: sp.Derivative, _graph_: nx.DiGraph, side_flag: int) -> Tuple[int, nx.DiGraph]:
        """
        Handles derivative expressions and adds them to the graph.

        Args:
            _expr_ (sp.Derivative): The derivative expression.
            _graph_ (nx.DiGraph): The graph to which the nodes and edges will be added.
            side_flag (int): Flag indicating side of the equation.

        Returns:
            Tuple[int, nx.DiGraph]: The node representing the derivative and the updated graph.
        """
        derivative_node = self.add_node(_graph_, side_flag, label=r'$ \hat{D} $', operator='Derivative')
        expr_node, _graph_ = self.process_expr(_expr_.args[0], _graph_, side_flag)
        self.add_edge(_graph_, expr_node, derivative_node)
        for iterm in _expr_.args[1:]:
            var_node, _graph_ = self.process_expr(iterm[0], _graph_, side_flag)
            self.add_edge(_graph_, derivative_node, var_node, variable=iterm[0])
        return derivative_node, _graph_

    def handle_integral(self, _expr_: sp.Integral, _graph_: nx.DiGraph, side_flag: int) -> Tuple[int, nx.DiGraph]:
        """
        Handles integral expressions and adds them to the graph.

        Args:
            _expr_ (sp.Integral): The integral expression.
            _graph_ (nx.DiGraph): The graph to which the nodes and edges will be added.
            side_flag (int): Flag indicating side of the equation.

        Returns:
            Tuple[int, nx.DiGraph]: The node representing the integral and the updated graph.
        """
        integral_node = self.add_node(_graph_, side_flag, label=r'$ \hat{I} $', operator='Integral')
        expr_node, _graph_ = self.process_expr(_expr_.args[0], _graph_, side_flag)
        self.add_edge(_graph_, expr_node, integral_node)
        for iterm in _expr_.args[1:]:
            var_node, _graph_ = self.process_expr(iterm[0], _graph_, side_flag)
            self.add_edge(_graph_, integral_node, var_node, variable=iterm[0])
        return integral_node, _graph_

    def handle_function(self, _expr_: sp.Expr, _graph_: nx.DiGraph, side_flag: int) -> Tuple[int, nx.DiGraph]:
        """
        Handles function expressions and adds them to the graph.

        Args:
            _expr_ (sp.Expr): The function expression.
            _graph_ (nx.DiGraph): The graph to which the nodes and edges will be added.
            side_flag (int): Flag indicating side of the equation.

        Returns:
            Tuple[int, nx.DiGraph]: The node representing the function and the updated graph.
        """
        func_node = self.add_node(_graph_, side_flag, label=str(_expr_.func), func=_expr_.func)
        for arg in _expr_.args:
            arg_node, _graph_ = self.process_expr(arg, _graph_, side_flag)
            self.add_edge(_graph_, arg_node, func_node)
        return func_node, _graph_

    def process_expr(self, _expr_: sp.Basic, _graph_: nx.DiGraph, side_flag: int) -> Tuple[int, nx.DiGraph]:
        """
        Processes a sympy expression and converts it to a NetworkX graph.

        Args:
            _expr_ (sp.Basic): The sympy expression to process.
            _graph_ (nx.DiGraph): The graph being generated.
            side_flag (int): Flag indicating side of the equation (-1: lhs, 1: rhs, 0: initial call).

        Returns:
            Tuple[int, nx.DiGraph]: The outermost node and the updated graph.
        """
        handler_map = {
            sp.Equality: self.handle_equality,
            sp.Add: self.handle_add,
            sp.Mul: self.handle_mul,
            sp.Pow: self.handle_pow,
            sp.Symbol: self.handle_symbol,
            sp.Integer: self.handle_number,
            sp.Float: self.handle_number,
            sp.Derivative: self.handle_derivative,
            sp.Integral: self.handle_integral,
            sp.Expr: self.handle_function
        }

        for expr_type, handler in handler_map.items():
            if isinstance(_expr_, expr_type):
                return handler(_expr_, _graph_, side_flag)

        raise ValueError(f"Unsupported expression type: {_expr_}")
