from itertools import chain, combinations, combinations_with_replacement, product
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sympy as sp  # Symbolic mathematics for manipulating expressions
from sympy.core.operations import AssocOp
from typing import Tuple, Union, Iterable


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
        self.imaginary_part = None
        self.reset_counters()

    def reset_counters(self):
        """
        Resets the counters for node naming based on the side of the expression.
        """
        self.node_count_pos = 0  # Counter for positive node names
        self.node_count_neg = 0  # Counter for negative node names

    def set_imaginary_part(self, imaginary_part: complex):
        """
        Sets the imaginary part to be used in node names.

        Args:
            imaginary_part (complex): The imaginary part for node naming.
        """
        self.imaginary_part = imaginary_part

    def get_next_node(self, side_flag: int) -> complex:
        """
        Determines the next node name based on the side of the expression.

        Args:
            side_flag (int): Flag indicating side of the equation (-1: lhs, 1: rhs, 0: initial call).

        Returns:
            str: The next node name.
        """
        if self.imaginary_part is None:
            return 0
        if side_flag == 0:
            return 0 + self.imaginary_part
        elif side_flag < 0:
            self.node_count_neg -= 1
            return self.node_count_neg + self.imaginary_part
        else:
            self.node_count_pos += 1
            return self.node_count_pos + self.imaginary_part

    @staticmethod
    def add_edge(graph: nx.DiGraph, from_node: complex, to_node: complex, **attrs):
        """
        Adds an edge to the graph.

        Args:
            graph (nx.DiGraph): The graph to which the edge will be added.
            from_node (str): The start node of the edge.
            to_node (str): The end node of the edge.
            **attrs: Additional attributes for the edge.
        """
        graph.add_edge(from_node, to_node, **attrs)

    def add_node(self, _graph_: nx.DiGraph, side_flag: int, label: str, **attrs) -> complex:
        """
        Adds a node to the graph with a unique name and label.

        Args:
            _graph_ (nx.DiGraph): The graph to which the node will be added.
            side_flag (int): Flag indicating side of the equation (-1: lhs, 1: rhs, 0: initial call).
            label (str): The label for the node.
            **attrs: Additional attributes for the node.

        Returns:
            complex: The name of the newly added node.
        """
        node_name = self.get_next_node(side_flag)
        #_graph_.add_node(node_name, label=f'{node_name}: {label}', **attrs)
        _graph_.add_node(node_name, label=f'{label}', **attrs)
        return node_name

    def process_iterable(self, expressions: Iterable[sp.Basic]) -> nx.DiGraph:
        """
        Processes an iterable of sympy expressions and converts them into a NetworkX graph.

        Args:
            expressions (Iterable[sp.Basic]): The iterable of sympy expressions to process.

        Returns:
            nx.DiGraph: The resulting graph.
        """
        graph = nx.DiGraph()
        central_node = self.add_node(graph, side_flag=0, label=r'$\exists$')
        if not np.iterable(expressions):
            expressions = [expressions, ]
        for expr_index, expr in enumerate(expressions):
            imaginary_part = (expr_index + 1) * 1j
            self.set_imaginary_part(imaginary_part)
            self.reset_counters()  # Reset counters for each new equation
            eq_node, graph = self.process_expr(expr, graph, side_flag=0)
            eq_name = f'Equation_{imaginary_part}j'
            self.add_edge(graph, eq_node, central_node, label=eq_name)
        return graph

    def process_expr(self, _expr_: sp.Basic, _graph_: nx.DiGraph, side_flag: int) -> Tuple[complex, nx.DiGraph]:
        """
        Processes a sympy expression and converts it to a NetworkX graph.

        Args:
            _expr_ (sp.Basic): The sympy expression to process.
            _graph_ (nx.DiGraph): The graph being generated.
            side_flag (int): Flag indicating side of the equation (-1: lhs, 1: rhs, 0: initial call).

        Returns:
            Tuple[str, nx.DiGraph]: The outermost node and the updated graph.
        """
        handler_map = {
            sp.Equality: self.handle_equality,
            sp.Add: self.handle_associative_operators,
            sp.Mul: self.handle_associative_operators,
            sp.Pow: self.handle_pow,
            sp.Symbol: self.handle_symbol,
            sp.Integer: self.handle_number,
            sp.Float: self.handle_number,
            sp.Derivative: self.handle_calculus_operators,
            sp.Integral: self.handle_calculus_operators,
            sp.Expr: self.handle_function
        }

        for expr_type, handler in handler_map.items():
            if isinstance(_expr_, expr_type):
                return handler(_expr_, _graph_, side_flag)

        raise ValueError(f"Unsupported expression type: {_expr_}")

    def handle_equality(self, _expr_: sp.Equality, _graph_: nx.DiGraph, side_flag: int) -> Tuple[complex, nx.DiGraph]:
        """
        Handles equality expressions and adds them to the graph.

        Args:
            _expr_ (sp.Equality): The equality expression.
            _graph_ (nx.DiGraph): The graph to which the nodes and edges will be added.
            side_flag (int): Flag indicating side of the equation (-1: lhs, 1: rhs).

        Returns:
            Tuple[str, nx.DiGraph]: The node representing the equality and the updated graph.
        """
        attributes = {'function': None, 'operator': '=', 'parameter': None, 'value': None, 'variable': None}
        eq_node = self.add_node(_graph_, side_flag=0, label='=', **attributes)
        lhs_node, _graph_ = self.process_expr(_expr_.lhs, _graph_, side_flag=-1)
        rhs_node, _graph_ = self.process_expr(_expr_.rhs, _graph_, side_flag=1)
        self.add_edge(_graph_, lhs_node, eq_node)
        self.add_edge(_graph_, rhs_node, eq_node)
        return eq_node, _graph_

    def handle_associative_operators(self, _expr_: AssocOp, _graph_: nx.DiGraph, side_flag: int) \
            -> Tuple[complex, nx.DiGraph]:
        """
        Handles addition and multiplication expressions and adds them to the graph.

        Args:
            _expr_ (AssocOp): The addition or multiplication expression.
            _graph_ (nx.DiGraph): The graph to which the nodes and edges will be added.
            side_flag (int): Flag indicating side of the equation.

        Returns:
            Tuple[str, nx.DiGraph]: The node representing the addition and the updated graph.
        """
        if isinstance(_expr_, sp.Add):
            _label_ = _operator_ = '+'
        else:
            _label_ = _operator_ = '*'
        attributes = {'function': _operator_, 'operator': _operator_, 'parameter': None, 'value': None,
                      'variable': None}
        operator_node = self.add_node(_graph_, side_flag, label=_label_, **attributes)
        for term in _expr_.args:
            term_node, _graph_ = self.process_expr(term, _graph_, side_flag)
            self.add_edge(_graph_, term_node, operator_node)
        return operator_node, _graph_

    def handle_pow(self, _expr_: sp.Pow, _graph_: nx.DiGraph, side_flag: int) -> Tuple[complex, nx.DiGraph]:
        """
        Handles power expressions and adds them to the graph.

        Args:
            _expr_ (sp.Pow): The power expression.
            _graph_ (nx.DiGraph): The graph to which the nodes and edges will be added.
            side_flag (int): Flag indicating side of the equation.

        Returns:
            Tuple[str, nx.DiGraph]: The node representing the power operation and the updated graph.
        """
        attributes = {'function': 'POW', 'operator': '**', 'parameter': None, 'value': None, 'variable': None}
        pow_node = self.add_node(_graph_, side_flag, label='**', **attributes)
        base_node, _graph_ = self.process_expr(_expr_.args[0], _graph_, side_flag)
        self.add_edge(_graph_, pow_node, base_node)
        exponent_node, _graph_ = self.process_expr(_expr_.args[1], _graph_, side_flag)
        self.add_edge(_graph_, exponent_node, pow_node)
        return pow_node, _graph_

    def handle_symbol(self, _expr_: sp.Symbol, _graph_: nx.DiGraph, side_flag: int) \
            -> Tuple[complex, nx.DiGraph]:
        """
        Handles symbol expressions and adds them to the graph.

        Args:
            _expr_ (sp.Symbol): The symbol expression.
            _graph_ (nx.DiGraph): The graph to which the nodes will be added.
            side_flag (int): Flag indicating side of the equation.

        Returns:
            Tuple[str, nx.DiGraph]: The node representing the symbol and the updated graph.
        """
        attributes = {'function': None, 'operator': _expr_, 'parameter': None, 'value': None, 'variable': _expr_}
        symbol_node = self.add_node(_graph_, side_flag, label=str(_expr_), **attributes)
        return symbol_node, _graph_

    def handle_number(self, _expr_: Union[sp.Integer, sp.Float], _graph_: nx.DiGraph, side_flag: int) \
            -> Tuple[complex, nx.DiGraph]:
        """
        Handles numeric expressions and adds them to the graph.

        Args:
            _expr_ (Union[sp.Integer, sp.Float]): The numeric expression.
            _graph_ (nx.DiGraph): The graph to which the nodes will be added.
            side_flag (int): Flag indicating side of the equation.

        Returns:
            Tuple[str, nx.DiGraph]: The node representing the number and the updated graph.
        """
        attributes = {'function': None, 'operator': _expr_, 'parameter': None, 'value': _expr_, 'variable': None}
        number_node = self.add_node(_graph_, side_flag, label=str(_expr_), **attributes)
        return number_node, _graph_

    def handle_calculus_operators(self, _expr_: Union[sp.Integral, sp.Derivative], _graph_: nx.DiGraph, side_flag: int) \
            -> Tuple[complex, nx.DiGraph]:
        """
        Handles integral and derivative expressions and adds them to the graph.

        Args:
            _expr_ (Union[sp.Integral, sp.Derivative]): The integral/derivative expression.
            _graph_ (nx.DiGraph): The graph to which the nodes and edges will be added.
            side_flag (int): Flag indicating side of the equation.

        Returns:
            Tuple[int, nx.DiGraph]: The node representing the integral/derivative and the updated graph.
        """
        if isinstance(_expr_, sp.Integral):
            _label_ = r'$ \hat{I} $'
            _operator_ = 'Integral'
        else:
            _label_ = r'$ \hat{D} $'
            _operator_ = 'Derivative'

        attributes = {'function': None, 'operator': _operator_, 'parameter': None, 'value': None, 'variable': None}
        calculus_node = self.add_node(_graph_, side_flag, label=_label_, **attributes)

        # Process the expression inside the integral/derivative
        expr_node, _graph_ = self.process_expr(_expr_.args[0], _graph_, side_flag)
        self.add_edge(_graph_, expr_node, calculus_node)

        # Process the variables (tuples) of the integral/derivative
        for var in _expr_.args[1:]:
            if len(var) == 1:
                num_iter = 0
            else:
                num_iter = var[1]
            for _iter_ in range(num_iter):
                var_node, _graph_ = self.process_expr(var[0], _graph_, side_flag)
                self.add_edge(_graph_, calculus_node, var_node, variable=var[0])
        return calculus_node, _graph_

    def handle_function(self, _expr_: sp.Expr, _graph_: nx.DiGraph, side_flag: int) -> Tuple[complex, nx.DiGraph]:
        """
        Handles general function expressions and adds them to the graph.

        Args:
            _expr_ (sp.Expr): The function expression.
            _graph_ (nx.DiGraph): The graph to which the nodes and edges will be added.
            side_flag (int): Flag indicating side of the equation.

        Returns:
            Tuple[str, nx.DiGraph]: The node representing the function and the updated graph.
        """
        attributes = {'function': type(_expr_), 'operator': type(_expr_), 'parameter': None, 'value': None,
                      'variable': None}
        func_node = self.add_node(_graph_, side_flag, label=str(type(_expr_).__name__), **attributes,
                                  func=_expr_.func)
        for arg in _expr_.args:
            arg_node, _graph_ = self.process_expr(arg, _graph_, side_flag)
            self.add_edge(_graph_, arg_node, func_node)
        return func_node, _graph_


class ExprGenerator:
    """
    A class to generate sympy expressions.
    """

    def __init__(self, sub_expressions: list = None):
        """
        Initialize the ExpressionGenerator.

        Args:
            sub_expressions (list): List of subexpressions to use in expressions.
        """
        if sub_expressions is None:
            sub_expressions = [sp.Symbol('_x_')]
        if hasattr(sub_expressions, '__iter__'):
            self.sub_expressions = np.array(sub_expressions)
        else:
            self.sub_expressions = np.array([sub_expressions])

        self._generate_polynomials()
        self._generate_trigonometric()
        self._generate_hyperbolic()
        self._generate_exponential()

    @staticmethod
    def _powerset(iterable):
        """
        Generate powerset from an iterable and return as a list

        Args:
            iterable: any iterable

        return:
            powerset (list): List of powerset of iterable
        """
        return [list(__) for __ in chain.from_iterable(
            combinations(iterable, sub_iterable) for sub_iterable in range(1, len(iterable) + 1))]

    def _generate_polynomials(self, max_degree: int = 3, all_terms: bool = False, cross_terms: bool = False):
        """
        Generate a list of sympy polynomial expressions

        Args:
            max_degree (int): Int of highest order polynomial term
            all_terms (bool): Boolean flagging whether to include lower order terms
            cross_terms (bool): Boolean flagging whether to include cross-terms
        """
        poly_list = []

        # Generate single variable polynomials
        for expr in self.sub_expressions:
            for degree in range(1, max_degree + 1):
                poly_list.append(expr ** degree)

        if all_terms:
            if cross_terms:
                # Generate cross terms without exceeding max_degree
                for degrees in product(range(max_degree + 1), repeat=len(self.sub_expressions)):
                    if 0 < sum(degrees) <= max_degree:
                        term = sp.Mul(*[expr ** deg for expr, deg in zip(self.sub_expressions, degrees)])
                        poly_list.append(term)
            else:
                for degree in range(1, max_degree + 1):
                    for terms in combinations_with_replacement(self.sub_expressions, degree):
                        poly_list.append(sp.Mul(*terms))

        self.poly_list = [sum(_expr_) for _expr_ in self._powerset(poly_list)]

    def _generate_trigonometric(self):
        """
        Generate a list of sympy trigonometric expressions
        """
        trig_func_list = [sp.sin, sp.cos, sp.tan, sp.csc, sp.sec, sp.cot]
        inv_trig_func_list = [sp.asin, sp.acos, sp.atan, sp.acsc, sp.asec, sp.acot]
        func_list = trig_func_list + inv_trig_func_list
        trig_list = []
        for _expr_ in self.sub_expressions:
            for func in func_list:
                trig_list.append(func(_expr_))
        self.trig_list = trig_list

    def _generate_hyperbolic(self):
        """
        Generate a list of sympy hyperbolic trigonometric expressions
        """
        trig_func_list = [sp.sinh, sp.cosh, sp.tanh, sp.csch, sp.sech, sp.coth]
        inv_trig_func_list = [sp.asinh, sp.acosh, sp.atanh, sp.acsch, sp.asech, sp.acoth]
        func_list = trig_func_list + inv_trig_func_list
        hyper_list = []
        for _expr_ in self.sub_expressions:
            for func in func_list:
                hyper_list.append(func(_expr_))
        self.hyper_list = hyper_list

    def _generate_exponential(self):
        """
        Generate a list of sympy exponential/logarithmic expressions
        """
        exp_list = []
        func_list = [sp.exp, sp.LambertW, sp.log]
        for _expr_ in self.sub_expressions:
            for func in func_list:
                exp_list.append(func(_expr_))
            for __expr__ in self.sub_expressions:
                exp_list.append(_expr_ ** __expr__)
            exp_list.append(2 ** _expr_)
        self.exp_list = exp_list

    def get_polynomials(self):
        """
        returns the base list of polynomials
        """
        return self.poly_list

    def get_trigonometric(self):
        """
        return the base list of trigonometric functions
        """
        return self.trig_list

    def get_hyperbolic(self):
        """
        return the base list of hyper trigonometric functions
        """
        return self.hyper_list

    def get_exponential(self):
        """
        return the base list of exponential/logarithmic functions
        """
        return self.exp_list


class GraphToExprConverter:
    """
    A class to convert a NetworkX graph back into sympy expressions.
    """

    def __init__(self):
        pass

    def graph_to_expr(self, graph: nx.DiGraph, node: complex) -> sp.Basic:
        """
        Converts a graph back into sympy expressions.

        Args:
            graph (nx.DiGraph): The graph representing the expressions.
            node (complex): The current node to process.

        Returns:
            sp.Basic: The corresponding sympy expression.
        """

        def associative_operator(_graph_: nx.DiGraph, _identity_: sp.Basic, _operator_: callable) -> sp.Basic:
            if len(predecessors) == 0:
                return _identity_
            terms = [self.graph_to_expr(_graph_, predecessor) for predecessor in predecessors]
            return _operator_(*terms)

        def calculus_operator(_graph_: nx.DiGraph, str_operator: str, _operator_: callable) -> sp.Basic:
            if len(predecessors) != 1:
                raise ValueError(f"{str_operator} node should have exactly 1 predecessor, got {len(predecessors)}")
            if len(successors) < 2:
                raise ValueError(f"{str_operator} node should have at least 2 successors, got {len(successors)}")
            expr_node = predecessors[0]
            variables = [successor for successor in successors if successor != min(successors, key=abs)]
            expr = self.graph_to_expr(_graph_, expr_node)
            vars_expressions = [self.graph_to_expr(_graph_, var_node) for var_node in variables]
            return _operator_(expr, *vars_expressions)

        def handle_equality() -> sp.Basic:
            if len(predecessors) != 2:
                raise ValueError(f"Equality node should have exactly 2 predecessors, got {len(predecessors)}")
            lhs = self.graph_to_expr(graph, predecessors[0])
            rhs = self.graph_to_expr(graph, predecessors[1])
            return sp.Eq(lhs, rhs)

        def handle_power() -> sp.Basic:
            if len(predecessors) != 1:
                raise ValueError(f"Power node should have exactly 1 predecessor, got {len(predecessors)}")
            if len(successors) != 2:
                raise ValueError(f"Power node should have exactly 2 successors, got {len(successors)}")
            base = self.graph_to_expr(graph, max(successors, key=abs))
            exp = self.graph_to_expr(graph, predecessors[0])
            return sp.Pow(base, exp)

        operator_handlers = {
            '=': handle_equality,
            '+': lambda: associative_operator(graph, sp.S.Zero, sp.Add),
            '*': lambda: associative_operator(graph, sp.S.One, sp.Mul),
            '**': handle_power,
            'Derivative': lambda: calculus_operator(graph, 'Derivative', sp.Derivative),
            'Integral': lambda: calculus_operator(graph, 'Integral', sp.Integral)
        }

        node_data: dict = graph.nodes[node]
        label: str = node_data.get('label', '')
        predecessors: list[complex] = list(graph.predecessors(node))
        successors: list[complex] = list(graph.successors(node))

        if node_data['operator'] in operator_handlers.keys():
            operator: str = node_data['operator']
            if operator in operator_handlers:
                return operator_handlers[operator]()
            else:
                raise ValueError(f"Unsupported operator: {operator}")

        elif node_data['parameter'] is not None:
            return sp.Symbol(label)

        elif node_data['variable'] is not None:
            return sp.Symbol(label)

        elif node_data['value'] is not None:
            return sp.Number(label)

        elif node_data['function'] is not None:
            if len(predecessors) != 1:
                raise ValueError(f"func node should have exactly 1 predecessor, got {len(predecessors)}")
            func: callable = node_data['func']
            return func(self.graph_to_expr(graph, predecessors[0]))

        else:
            raise ValueError(f"Unsupported node data: {node_data}")

    def convert_graph(self, graph: nx.DiGraph) -> list[sp.Basic]:
        """
        Converts the entire graph containing multiple equations back into a list of sympy expressions.

        Args:
            graph (nx.DiGraph): The graph representing the expressions.

        Returns:
            list[sp.Basic]: The list of corresponding sympy expressions.
        """
        central_node = 0
        eq_nodes = list(graph.predecessors(central_node))
        expressions = []
        for eq_node in eq_nodes:
            expr = self.graph_to_expr(graph, eq_node)
            expressions.append(expr)
        return expressions


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
