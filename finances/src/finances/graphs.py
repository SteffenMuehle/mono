import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import toml
from graphviz import Digraph


class Node:
    def __init__(
        self,
        id: str,
        print_name: str | None = None,
        parent_id: str | None = None,
        current_amount: float | None = None,
        target_amount: float | None = None,
        weight: float | None = None,
    ):
        self.id = id
        if print_name is None:
            self.print_name = id
        else:
            self.print_name = print_name
        self.parent_id = parent_id
        self.children_ids: list[str] = []
        self.current_amount = current_amount
        self.target_amount = target_amount
        self.frozen = self.target_amount is not None
        self.weight = weight

    def __repr__(self):
        return f"Node(id = '{self.id}', parent_id='{self.parent_id}', children='{self.children_ids}'"


class Graph:
    def __init__(
        self,
        nodes: list[Node],
    ):
        self.nodes_dict = {node.id: node for node in nodes}
        self._equilibrate()
        assert all(key == val.id for key, val in self.nodes_dict.items())

    def _assign_all_children_to_their_parents(self):
        for this_id, this_node in self.nodes_dict.items():
            this_node.children_ids = [other_id for other_id, other_node in self.nodes_dict.items() if other_node.parent_id == this_id]

    def __repr__(self):
        return_str = "Graph(node_ids=[\n"
        for node_id in self.nodes_dict:
            return_str += f"{node_id},\n"
        return_str += "])"
        return return_str

    def _get_node_by_id(self, node_id: str) -> Node:
        return self.nodes_dict[node_id]

    def _get_children_of_node_by_id(self, parent_node_id: str) -> list[Node]:
        parent_node = self._get_node_by_id(parent_node_id)
        return [self._get_node_by_id(child_id) for child_id in parent_node.children_ids]

    def _get_root_of_graph(self) -> Node:
        roots = [node for node in self.nodes if node.parent_id is None]
        num_roots = len(roots)
        if num_roots == 0:
            raise ValueError("No root nodes found")
        if num_roots > 1:
            raise ValueError("Multiple root nodes found")
        return roots[0]

    def _get_leaves(self) -> list[Node]:
        return [node for node in self.nodes if not node.children]

    def _propagate_current_amounts_up(self):
        def _get_current_amount_by_id(node_id: str) -> float:
            children = self._get_children_of_node_by_id(node_id)
            if not children:
                to_return = self._get_node_by_id(node_id).current_amount
            else:
                to_return = sum([_get_current_amount_by_id(child.id) for child in children])
                self._get_node_by_id(node_id).current_amount = to_return
            return to_return

        _get_current_amount_by_id(self._get_root_of_graph().id)

    def _propagate_target_amounts_down(self):
        root_node = self._get_root_of_graph()
        root_node.target_amount = root_node.current_amount
        current_parent_ids = [root_node.id]
        while current_parent_ids:
            new_parent_ids = []
            for parent_id in current_parent_ids:
                parent_node = self._get_node_by_id(parent_id)
                amount_to_inherit = parent_node.target_amount
                children = self._get_children_of_node_by_id(parent_id)
                if not children:
                    continue

                # frozen children just keep their target_amount and don't further inherit from parent:
                # remove frozen children's target_amount from parent's inheritable amount
                frozen_children = [child for child in children if child.frozen]
                if frozen_children:
                    frozen_target_amount = sum([child.target_amount for child in frozen_children])
                    amount_to_inherit -= frozen_target_amount

                # distribute remaining target_amount to remaining children
                inheritting_children = [child for child in children if not child.frozen]
                combined_weight = sum([child.weight for child in inheritting_children])
                for child in inheritting_children:
                    child.target_amount = amount_to_inherit * child.weight / combined_weight
                    new_parent_ids.append(child.id)
            current_parent_ids = new_parent_ids

    def _propagate_frozen_status_up_one_step(self):
        for node_id, node in self.nodes.items():
            children = self._get_children_of_node_by_id(node_id)
            if children and all([child.frozen for child in children]):
                node.target_amount = sum([child.target_amount for child in children])
                node.frozen = True

    def _propagate_frozen_status_up(self):
        def _num_frozen_nodes_in_graph(self):
            return sum([node.frozen for _, node in self.nodes.items()])
        num_frozen_nodes_before = 0
        num_frozen_nodes_after = _num_frozen_nodes_in_graph(self)
        while num_frozen_nodes_before != num_frozen_nodes_after:
            num_frozen_nodes_before = num_frozen_nodes_after
            self._propagate_frozen_status_up_one_step()
            num_frozen_nodes_after = _num_frozen_nodes_in_graph(self)

    def _equilibrate(self):
        self._assign_all_children_to_their_parents()
        self._propagate_current_amounts_up()
        self._propagate_frozen_status_up()
        self._propagate_target_amounts_down()

    def plot_graph(self, output_path: str = "graph.svg"):
        graph = Digraph()

        total_amount = self._get_root_of_graph().current_amount
        min_font_size = 8
        max_font_size = 16
        for node in self.nodes:
            color = "lightblue" if node.frozen else "orange"
            # Customize node appearance
            graph.node(
                node.id,
                label=node.print_name + f"\n {int(round(node.current_amount, ndigits=0))}",
                shape="box",
                style="filled",
                color=color,
                fontname="Helvetica",
                fontsize="10",
            )
            if node.parent:
                fraction = node.current_amount / total_amount
                curr_font_size = max(min_font_size, max_font_size * np.sqrt(fraction))
                # Customize edge appearance
                graph.edge(
                    node.parent,
                    node.id,
                    color="black",
                    # weight=str(node.target_amount),
                    label=(str(int(round(100 * node.weight, ndigits=0))) + "%" if not node.frozen else "fixed")
                    + " = "
                    + str(int(round(node.target_amount, ndigits=0))),
                    fontsize=str(int(curr_font_size)),
                )
        return graph.render(output_path, format="png", cleanup=True)

    def to_markdown(self):
        def _to_markdown_recursive(node_id: str, depth: int):
            node = self._get_node_by_id(node_id)
            children = self._get_children_of_node_by_id(node_id)
            indent = "  " * depth
            if not children:
                return indent + f"- {node.print_name} ({node.id}): {round(node.current_amount, 2)}\n"
            else:
                return (
                    indent
                    + f"- {node.print_name} (total: {round(node.current_amount, 2)})\n"
                    + "".join([_to_markdown_recursive(child.id, depth + 1) for child in children])
                )

        return _to_markdown_recursive(self._get_root_of_graph().id, 0)

    def absorb_graph(
        self,
        graph_that_is_absorbed,
        node_id_to_attach_to: str = "",
    ):
        sub_root_id = graph_that_is_absorbed._get_root().id

        self.nodes += graph_that_is_absorbed.nodes
        self._get_node_by_id(sub_root_id).parent_id = node_id_to_attach_to
        self._equilibrate()

    def save_to_csv(
        self,
        csv_path: str,
    ):
        csv_columns = [
            "id",
            "print_name",
            "date",
            "current_amount",
            "target_amount",
        ]
        # write to csv:
        # # find out if csv file already exists
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a") as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns)
            if not file_exists:
                writer.writeheader()
            for node in self.nodes:
                writer.writerow(
                    {
                        "id": node.id,
                        "print_name": node.print_name,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "current_amount": round(node.current_amount, 2),
                        "target_amount": round(node.target_amount, 2),
                    }
                )

    @classmethod
    def from_toml(cls, toml_path: str):
        def parse_nodes(data, parent_id=None):
            nodes = []
            for key, val in data.items():
                if isinstance(val, dict):
                    node_id = key.split(".")[-1]
                    cur_node = Node(
                        id=node_id,
                        print_name=val.get("print_name", None),
                        parent_id=parent_id,
                        current_amount=val.get("current_amount", None),
                        target_amount=val.get("target_amount", None),
                        weight=val.get("weight", 1.0),
                    )
                    nodes.append(cur_node)
                    nodes.extend(parse_nodes(val, parent_id=node_id))
            return nodes

        with open(toml_path) as file:
            data = toml.load(file)

        nodes = parse_nodes(data)
        return cls(nodes=nodes)


if __name__ == "__main__":
    input_path = Path(__file__).parent.parent.parent / "data" / "input" / "graph"
    output_path = Path(__file__).parent.parent.parent / "data" / "output"
    today = datetime.now().strftime("%Y-%m-%d")

    # read graphs
    base_graph = Graph.from_toml(input_path / "main.md")
    elisa_graph = Graph.from_toml(input_path / "elisa.md")
    set_aside_graph = Graph.from_toml(input_path / "set_aside.md")
    investment_graph = Graph.from_toml(input_path / "investments.md")
    crypto_graph = Graph.from_toml(input_path / "crypto.md")

    # combine graphs
    base_graph.absorb_graph(elisa_graph)
    base_graph.absorb_graph(
        set_aside_graph,
        node_id_to_attach_to="root",
    )
    base_graph.absorb_graph(
        crypto_graph,
        node_id_to_attach_to="root",
    )
    base_graph.absorb_graph(
        investment_graph,
        node_id_to_attach_to="Steffen",
    )

    # plot result
    base_graph.plot_graph(output_path / today / "graph")  # inner function appends .png suffix

    # save graph to csv
    base_graph.save_to_csv(output_path / "graph_history.csv")

    # print markdown to stdout, terminal caller can pipe it into a file
    print("\n# Portfolio")
    print(base_graph.to_markdown())
