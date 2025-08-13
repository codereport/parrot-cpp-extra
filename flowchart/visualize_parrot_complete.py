#!/usr/bin/env python3
"""
Complete Parrot Methods Dependency Visualization using Graphviz
Generates a comprehensive visual graph showing all individual Parrot methods
and their underlying Thrust algorithm and fancy iterator dependencies.
"""

import graphviz
from typing import Dict, List, Tuple


def create_complete_dependency_graph() -> graphviz.Digraph:
    """Create a comprehensive graph showing all Parrot methods and dependencies."""

    # Create the main directed graph with hierarchical layout
    dot = graphviz.Digraph(comment="Complete Parrot Methods Dependencies")
    dot.attr(rankdir="TB", size="30,24", dpi="300")
    dot.attr("node", style="filled", fontname="Arial", fontsize="9")
    dot.attr("edge", fontname="Arial", fontsize="7")

    # Define color schemes
    colors = {
        "star_method": "#fff9c4",  # Yellow - Star methods (⭐)
        "method_node": "#e1f5fe",  # Light Blue - Regular methods
        "thrust_algo": "#fff3e0",  # Orange - Thrust algorithms
        "thrust_iter": "#f3e5f5",  # Purple - Thrust iterators
        "cub_algo": "#ffebee",  # Red - CUB algorithms
    }

    # Core Thrust Iterators
    with dot.subgraph(name="cluster_thrust_iterators") as c:
        c.attr(label="Thrust Fancy Iterators", style="filled", color="lightgrey")
        iterators = [
            ("TI", "transform_iterator"),
            ("ZI", "zip_iterator"),
            ("CI", "counting_iterator"),
            ("PI", "permutation_iterator"),
            ("CONST_I", "constant_iterator"),
            ("REV_I", "reverse_iterator"),
            ("DISCARD_I", "discard_iterator"),
        ]
        for node_id, label in iterators:
            c.node(node_id, label, color=colors["thrust_iter"], shape="box")

    # Thrust Algorithms
    with dot.subgraph(name="cluster_thrust_algorithms") as c:
        c.attr(label="Thrust Algorithms", style="filled", color="lightgrey")
        algorithms = [
            ("T_REDUCE", "thrust::reduce"),
            ("T_SCAN", "thrust::inclusive_scan"),
            ("T_SCAN_BY_KEY", "thrust::inclusive_scan_by_key"),
            ("T_SORT", "thrust::sort"),
            ("T_COPY", "thrust::copy"),
            ("T_COPY_IF", "thrust::copy_if"),
            ("T_COUNT_IF", "thrust::count_if"),
            ("T_EQUAL", "thrust::equal"),
            ("T_DISTANCE", "thrust::distance"),
            ("T_REDUCE_BY_KEY", "thrust::reduce_by_key"),
            ("T_MAX_ELEMENT", "thrust::max_element"),
            ("T_TRANSFORM", "thrust::transform"),
            ("T_ZIP_FUNCTION", "thrust::zip_function"),
        ]
        for node_id, label in algorithms:
            c.node(node_id, label, color=colors["thrust_algo"], shape="box")

    # CUB Algorithms
    with dot.subgraph(name="cluster_cub") as c:
        c.attr(label="CUB Algorithms", style="filled", color="lightgrey")
        c.node(
            "CUB_REDUCE",
            "cub::DeviceSegmentedReduce",
            color=colors["cub_algo"],
            shape="box",
        )

    # Note: Removed Core Concepts cluster per user request

    # Lazy Operations - 1-index Maps (Unary)
    with dot.subgraph(name="cluster_unary_maps") as c:
        c.attr(label="1-index Maps (Unary)", style="filled", color="#f0f8ff")
        unary_methods = [
            ("MAP", "⭐ map", True),  # Star method
            ("ABS", "abs", False),
            ("DBLE", "dble", False),
            ("EVEN", "even", False),
            ("HALF", "half", False),
            ("LOG", "log", False),
            ("EXP", "exp", False),
            ("NEG", "neg", False),
            ("ODD", "odd", False),
            ("RAND", "rand", False),
            ("SQ", "sq", False),
            ("SQRT", "sqrt", False),
        ]
        for node_id, label, is_star in unary_methods:
            color = colors["star_method"] if is_star else colors["method_node"]
            c.node(node_id, label, color=color, shape="ellipse")

    # Lazy Operations - 1-index Maps (Binary)
    with dot.subgraph(name="cluster_binary_maps") as c:
        c.attr(label="1-index Maps (Binary)", style="filled", color="#f0f8ff")
        binary_methods = [
            ("MAP2", "⭐ map2", True),  # Star method
            ("ADD", "add (+)", False),
            ("DIV", "div (/)", False),
            ("GT", "gt (>)", False),
            ("GTE", "gte (>=)", False),
            ("IDIV", "idiv", False),
            ("LT", "lt (<)", False),
            ("LTE", "lte (<=)", False),
            ("MAX_OP", "max", False),
            ("MIN_OP", "min", False),
            ("MINUS", "minus (-)", False),
            ("TIMES", "times (*)", False),
            ("PAIRS", "pairs", False),
        ]
        for node_id, label, is_star in binary_methods:
            color = colors["star_method"] if is_star else colors["method_node"]
            c.node(node_id, label, color=color, shape="ellipse")

    # Lazy Operations - 2-index Maps
    with dot.subgraph(name="cluster_2index_maps") as c:
        c.attr(label="2-index Maps", style="filled", color="#f0f8ff")
        methods = [
            ("MAP_ADJ", "⭐ map_adj", True),  # Star method
            ("DELTAS", "deltas", False),
            ("DIFFER", "differ", False),
        ]
        for node_id, label, is_star in methods:
            color = colors["star_method"] if is_star else colors["method_node"]
            c.node(node_id, label, color=color, shape="ellipse")

    # Eager Operations - Reductions
    with dot.subgraph(name="cluster_reductions") as c:
        c.attr(label="Reductions", style="filled", color="#ffe0e0")
        reduction_methods = [
            ("REDUCE", "⭐ reduce", True),  # Star method
            ("ALL", "all", False),
            ("ANY", "any", False),
            ("MAXR", "maxr", False),
            ("MAX_BY_KEY", "max_by_key", False),
            ("MINR", "minr", False),
            ("MINMAX", "minmax", False),
            ("PROD", "prod", False),
            ("SUM", "sum", False),
        ]
        for node_id, label, is_star in reduction_methods:
            color = colors["star_method"] if is_star else colors["method_node"]
            c.node(node_id, label, color=color, shape="ellipse")

    # Eager Operations - Scans
    with dot.subgraph(name="cluster_scans") as c:
        c.attr(label="Scans", style="filled", color="#ffe0e0")
        scan_methods = [
            ("SCAN", "⭐ scan", True),  # Star method
            ("ALLS", "alls", False),
            ("ANYS", "anys", False),
            ("MAXS", "maxs", False),
            ("MINS", "mins", False),
            ("PRODS", "prods", False),
            ("SUMS", "sums", False),
        ]
        for node_id, label, is_star in scan_methods:
            color = colors["star_method"] if is_star else colors["method_node"]
            c.node(node_id, label, color=color, shape="ellipse")

    # Lazy-ish Operations - Compactions
    with dot.subgraph(name="cluster_compactions") as c:
        c.attr(label="Compactions", style="filled", color="#fff0e0")
        compaction_methods = [
            ("KEEP", "⭐ keep", True),  # Star method
            ("FILTER", "filter", False),
            ("WHERE", "where", False),
            ("UNIQ", "uniq", False),
            ("DISTINCT", "distinct", False),
        ]
        for node_id, label, is_star in compaction_methods:
            color = colors["star_method"] if is_star else colors["method_node"]
            c.node(node_id, label, color=color, shape="ellipse")

    # Additional method groups
    other_methods = [
        # Reshapes
        ("TAKE", "take"),
        ("DROP", "drop"),
        ("TRANSPOSE", "transpose"),
        ("RESHAPE", "reshape"),
        ("CYCLE", "cycle"),
        ("REPEAT", "repeat"),
        ("REPLICATE", "replicate"),
        # Joins
        ("APPEND", "append"),
        ("PREPEND", "prepend"),
        # Products
        ("CROSS", "cross"),
        ("OUTER", "outer"),
        # Permutations
        ("REV", "rev"),
        ("GATHER", "gather"),
        # Sorting
        ("SORT", "sort"),
        ("SORT_BY", "sort_by"),
        ("SORT_BY_KEY", "sort_by_key"),
        # Other
        ("RLE", "rle"),
        ("CHUNK_BY_REDUCE", "chunk_by_reduce"),
        ("MATCH", "match"),
        ("SIZE", "size"),
        ("TO_HOST", "to_host"),
        # Array Creation
        ("ARRAY", "array"),
        ("RANGE", "range"),
        ("SCALAR", "scalar"),
        ("MATRIX", "matrix"),
        # Statistical
        ("NORM_CDF", "norm_cdf"),
    ]

    for node_id, label in other_methods:
        dot.node(node_id, label, color=colors["method_node"], shape="ellipse")

    # Define key dependencies (simplified for readability)
    dependencies = [
        # Core map operations
        ("MAP", "TI"),
        ("MAP2", "TI"),
        ("MAP2", "ZI"),
        # Unary maps depend on map
        ("ABS", "MAP"),
        ("DBLE", "MAP"),
        ("EVEN", "MAP"),
        ("HALF", "MAP"),
        ("LOG", "MAP"),
        ("EXP", "MAP"),
        ("NEG", "MAP"),
        ("ODD", "MAP"),
        ("SQ", "MAP"),
        ("SQRT", "MAP"),
        # Binary maps depend on map2
        ("ADD", "MAP2"),
        ("DIV", "MAP2"),
        ("GT", "MAP2"),
        ("GTE", "MAP2"),
        ("IDIV", "MAP2"),
        ("LT", "MAP2"),
        ("LTE", "MAP2"),
        ("MAX_OP", "MAP2"),
        ("MIN_OP", "MAP2"),
        ("MINUS", "MAP2"),
        ("TIMES", "MAP2"),
        # 2-index maps
        ("MAP_ADJ", "TI"),
        ("MAP_ADJ", "ZI"),
        ("MAP_ADJ", "T_ZIP_FUNCTION"),
        ("DELTAS", "MAP_ADJ"),
        ("DIFFER", "MAP_ADJ"),
        # Reductions
        ("REDUCE", "T_REDUCE"),
        ("REDUCE", "CUB_REDUCE"),
        ("ALL", "REDUCE"),
        ("ANY", "REDUCE"),
        ("MAXR", "REDUCE"),
        ("MINR", "REDUCE"),
        ("MINMAX", "REDUCE"),
        ("PROD", "REDUCE"),
        ("SUM", "REDUCE"),
        ("MAX_BY_KEY", "T_MAX_ELEMENT"),
        # Scans
        ("SCAN", "T_SCAN"),
        ("SCAN", "T_SCAN_BY_KEY"),
        ("ALLS", "SCAN"),
        ("ANYS", "SCAN"),
        ("MAXS", "SCAN"),
        ("MINS", "SCAN"),
        ("PRODS", "SCAN"),
        ("SUMS", "SCAN"),
        # Compactions
        ("KEEP", "T_COPY_IF"),
        ("FILTER", "TI"),
        ("FILTER", "KEEP"),
        ("WHERE", "RANGE"),
        ("WHERE", "KEEP"),
        ("UNIQ", "KEEP"),
        ("UNIQ", "DIFFER"),
        ("UNIQ", "PREPEND"),
        ("DISTINCT", "SORT"),
        ("DISTINCT", "UNIQ"),
        # Products point to other Parrot methods
        ("CROSS", "REPLICATE"),
        ("CROSS", "CYCLE"),
        ("CROSS", "PAIRS"),
        ("OUTER", "TI"),
        ("OUTER", "CI"),
        # Matrix creation points to other Parrot methods
        ("MATRIX", "SCALAR"),
        ("MATRIX", "REPEAT"),
        ("MATRIX", "RESHAPE"),
        # Other key dependencies
        ("TRANSPOSE", "PI"),
        ("TRANSPOSE", "TI"),
        ("TRANSPOSE", "CI"),
        ("GATHER", "PI"),
        ("REV", "REV_I"),
        ("SORT", "T_SORT"),
        ("SORT", "T_COPY"),
        ("SORT_BY", "T_SORT"),
        ("SORT_BY", "T_COPY"),
        ("SORT_BY_KEY", "T_SORT"),
        ("SORT_BY_KEY", "T_COPY"),
        ("RLE", "T_REDUCE_BY_KEY"),
        ("RLE", "CONST_I"),
        ("RLE", "ZI"),
        ("RLE", "TI"),
        ("CHUNK_BY_REDUCE", "T_REDUCE_BY_KEY"),
        ("CHUNK_BY_REDUCE", "DISCARD_I"),
        ("PAIRS", "ZI"),
        ("PAIRS", "TI"),
        ("RAND", "TI"),
        ("RAND", "ZI"),
        ("RAND", "CI"),
        ("APPEND", "TI"),
        ("APPEND", "CI"),
        ("PREPEND", "TI"),
        ("PREPEND", "CI"),
        ("CYCLE", "TI"),
        ("CYCLE", "CI"),
        ("REPEAT", "CONST_I"),
        ("REPLICATE", "TI"),
        ("REPLICATE", "CI"),
        ("TAKE", "TI"),
        ("DROP", "TI"),
        ("RESHAPE", "TI"),
        ("SIZE", "T_DISTANCE"),
        ("SIZE", "T_COUNT_IF"),
        ("TO_HOST", "T_COPY"),
        ("MATCH", "T_EQUAL"),
        ("RANGE", "CI"),
        ("SCALAR", "CONST_I"),
        ("ARRAY", "T_COPY"),
        ("NORM_CDF", "MAP"),
    ]

    # Add edges
    for source, target in dependencies:
        dot.edge(source, target)

    return dot


def main():
    """Generate the complete visualization."""

    print("Generating complete Parrot dependency visualization...")

    # Create the complete dependency graph
    complete_graph = create_complete_dependency_graph()
    complete_graph.render("parrot_complete_dependencies", format="png", cleanup=True)
    complete_graph.render("parrot_complete_dependencies", format="svg", cleanup=True)
    complete_graph.render("parrot_complete_dependencies", format="pdf", cleanup=True)

    print("Generated files:")
    print("  - parrot_complete_dependencies.png/svg/pdf")
    print("\nTo install graphviz:")
    print("  pip install graphviz")
    print(
        "  # Also need system graphviz: apt-get install graphviz (Ubuntu) or brew install graphviz (Mac)"
    )


if __name__ == "__main__":
    main()
