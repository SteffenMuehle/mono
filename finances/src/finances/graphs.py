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
        inheritance_weight: float | None = None,
    ):
        self.id = id
        self.print_name = print_name if print_name is not None else id
        self.parent_id = parent_id
        self.current_amount = current_amount
        self.target_amount = target_amount
        self.inheritance_weight = inheritance_weight
        self.children_ids: list[str] = []
        self.plotcolor = "orange"  # default color for non-frozen nodes

    @property
    def frozen(self) -> bool:
        return self.target_amount is not None

    def __repr__(self):
        return f"Node(id = '{self.id}', parent_id='{self.parent_id}', children='{self.children_ids}'"


class Graph:
    def __init__(
        self,
        nodes: list[Node],
    ):
        self.nodes_dict = {node.id: node for node in nodes}

    def _assign_all_children_to_their_parents(self):
        for this_id, this_node in self.nodes_dict.items():
            this_node.children_ids = [
                other_id for other_id, other_node in self.nodes_dict.items() if other_node.parent_id == this_id
            ]

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
        roots = [node for _, node in self.nodes_dict.items() if node.parent_id == "root"]
        num_roots = len(roots)
        if num_roots == 0:
            raise ValueError("No root nodes found")
        if num_roots > 1:
            raise ValueError("Multiple root nodes found")
        return roots[0]

    def _propagate_current_amounts_up(self):
        def _get_current_amount_by_id(node_id: str) -> float:
            children = self._get_children_of_node_by_id(node_id)
            node = self._get_node_by_id(node_id)
            if not children:
                to_return = node.current_amount
            else:
                to_return = sum([_get_current_amount_by_id(child.id) for child in children])
                if node.current_amount is not None:
                    raise ValueError(
                        f"Node {node_id} has a current_amount set ({node.current_amount}) but also has children ({children}) with total current amount {to_return}."
                        "This is not allowed, as it would be ambiguous how to propagate the current_amount up."
                    )
                else:
                    node.current_amount = to_return
            return to_return  # type: ignore

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
                combined_weight = sum([child.inheritance_weight for child in inheritting_children])
                for child in inheritting_children:
                    child.target_amount = amount_to_inherit * child.inheritance_weight / combined_weight
                    new_parent_ids.append(child.id)
            current_parent_ids = new_parent_ids

    def _propagate_frozen_status_up_one_step(self):
        for node_id, node in self.nodes_dict.items():
            children = self._get_children_of_node_by_id(node_id)
            if children and all([child.frozen for child in children]):
                node.target_amount = sum([child.target_amount for child in children])

    def _propagate_frozen_status_up(self):
        def _num_frozen_nodes_in_graph(self):
            return sum([node.frozen for _, node in self.nodes_dict.items()])

        num_frozen_nodes_before = 0
        num_frozen_nodes_after = _num_frozen_nodes_in_graph(self)
        while num_frozen_nodes_before != num_frozen_nodes_after:
            num_frozen_nodes_before = num_frozen_nodes_after
            self._propagate_frozen_status_up_one_step()
            num_frozen_nodes_after = _num_frozen_nodes_in_graph(self)

    def _check_validity_of_nodes(self):
        for node_id, node in self.nodes_dict.items():
            if node.parent_id == "root":
                if node.inheritance_weight is not None:
                    raise ValueError(f"root node {node_id} cannot have a weight set, as it has noone to inherit from.")
            elif node.frozen:
                if node.inheritance_weight is not None:
                    raise ValueError(
                        f"Node {node_id} has a target_amount ({node.target_amount}) set."
                        f"Thus it will not inherit from its parent ({node.parent_id}) and we don't its weight to be set but it is {node.inheritance_weight}."
                    )
            elif node.inheritance_weight is None:
                raise ValueError(
                    f"Node {node_id} is not frozen and is expected to inherit from its parent ({node.parent_id}) but has no weight set."
                    f"Please set the weight to a value between 0 and 1."
                )
            elif node.inheritance_weight < 0 or node.inheritance_weight > 1:
                raise ValueError(f"Node {node_id} has a weight set ({node.inheritance_weight}) but it is not in [0,1].")

    def _fixate_plotcolors(self):
        for _, node in self.nodes_dict.items():
            if node.frozen:
                node.plotcolor = "lightblue"
            else:
                node.plotcolor = "orange"
            if node.parent_id == "root":
                node.plotcolor = "lightgreen"
            elif node.parent_id == "disconnected":
                node.plotcolor = "red"

    def equilibrate(self):
        self._assign_all_children_to_their_parents()
        self._propagate_frozen_status_up()
        self._check_validity_of_nodes()
        self._propagate_current_amounts_up()
        self._fixate_plotcolors()
        self._propagate_target_amounts_down()

    def plot_graph(self, output_path: str = "graph.svg"):
        graph = Digraph()

        total_amount = self._get_root_of_graph().current_amount
        if total_amount is None:
            raise ValueError("Root node has no current_amount set.")
        min_font_size = 8
        max_font_size = 16
        for _, node in self.nodes_dict.items():
            current_amount = node.current_amount
            if current_amount is None:
                raise ValueError(f"Node {node.id} has no current_amount set.")
            target_amount = node.target_amount
            if target_amount is None:
                raise ValueError(f"Node {node.id} has no target_amount set.")
            # Customize node appearance
            graph.node(
                node.id,
                label=node.print_name + f"\n {int(round(current_amount, ndigits=0))}",  # type: ignore
                shape="box",
                style="filled",
                color=node.plotcolor,
                fontname="Helvetica",
                fontsize="10",
            )
            if node.parent_id and node.parent_id not in ["root", "disconnected"]:
                fraction = current_amount / total_amount
                curr_font_size = max(min_font_size, max_font_size * np.sqrt(fraction))
                # Customize edge appearance
                graph.edge(
                    node.parent_id,
                    node.id,
                    color="black",
                    label=(
                        str(int(round(100 * node.inheritance_weight, ndigits=0))) + "%"  # type: ignore
                        if node.plotcolor == "orange"
                        else "fixed"
                    )
                    + " = "
                    + str(int(round(target_amount, ndigits=0))),
                    fontsize=str(int(curr_font_size)),
                )
        return graph.render(output_path, format="png", cleanup=True)

    def to_markdown(self):
        def _to_markdown_recursive(node_id: str, depth: int):
            node = self._get_node_by_id(node_id)
            children = self._get_children_of_node_by_id(node_id)
            indent = "  " * depth
            if not children:
                return indent + f"- {node.print_name} ({node.id}): {round(node.current_amount, 2)}\n"  # type: ignore
            else:
                return (
                    indent
                    + f"- {node.print_name} (total: {round(node.current_amount, 2)})\n"  # type: ignore
                    + "".join([_to_markdown_recursive(child.id, depth + 1) for child in children])
                )

        return _to_markdown_recursive(self._get_root_of_graph().id, 0)

    def absorb_graph(
        self,
        graph_that_is_absorbed: "Graph",
        node_id_to_attach_to: str = "disconnected",
    ):
        sub_root_id = graph_that_is_absorbed._get_root_of_graph().id

        self.nodes_dict |= graph_that_is_absorbed.nodes_dict
        self._get_node_by_id(sub_root_id).parent_id = node_id_to_attach_to

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
            for _, node in self.nodes_dict.items():
                writer.writerow(
                    {
                        "id": node.id,
                        "print_name": node.print_name,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "current_amount": round(node.current_amount, 2),  # type: ignore
                        "target_amount": round(node.target_amount, 2),  # type: ignore
                    }
                )

    @classmethod
    def from_toml(cls, toml_path: Path):
        def parse_nodes(data, parent_id="root"):
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
                        inheritance_weight=val.get("weight", None),
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

    # combine graphs
    base_graph.absorb_graph(elisa_graph)
    base_graph.absorb_graph(
        set_aside_graph,
        node_id_to_attach_to="total",
    )
    base_graph.absorb_graph(
        investment_graph,
        node_id_to_attach_to="Steffen",
    )
    base_graph.equilibrate()

    # plot result
    base_graph.plot_graph(output_path / today / "graph")  # inner function appends .png suffix

    # save graph to csv
    base_graph.save_to_csv(output_path / "graph_history.csv")

    # print markdown to stdout, terminal caller can pipe it into a file
    print("\n# Portfolio")
    print(base_graph.to_markdown())
