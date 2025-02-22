from typing import Optional
from graphviz import Digraph
import toml
import os
import csv
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
from collections import defaultdict
import numpy as np

class Node:
    def __init__(
        self,
        id: str,
        print_name: Optional[str] = None,
        parent: Optional[str] = None,
        current_amount: Optional[float] = None,
        target_amount: Optional[float] = None,
        weight: Optional[float] = None,
    ):
        self.id = id
        if print_name is None:
            self.print_name = id
        else:
            self.print_name = print_name
        self.parent = parent
        self.children = None
        self.current_amount = current_amount
        self.target_amount = target_amount
        self.frozen = self.target_amount is not None
        self.weight = weight

    def __repr__(self):
        return f"Node(id = '{self.id}', parent='{self.parent}', children='{self.children}'"


class Graph:
    def __init__(
        self,
        nodes: list[Node],
    ):
        self.nodes = nodes
        # fill in nodes' children
        self._equilibrate()

    def _find_children(self):
        for node in self.nodes:
            node.children = [other_node.id for other_node in self.nodes if other_node.parent == node.id]


    def __repr__(self):
        return_str = "Graph(nodes=[\n"
        for node in self.nodes:
            return_str += f"{node},\n"
        return_str += "])"
        return return_str

    def _get_node(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise ValueError(f"Node with id {node_id} not found")

    def _get_children(self, node_id: str) -> list[Node]:
        node = self._get_node(node_id)
        return node.children

    def _get_root(self) -> Node:
        roots = [node for node in self.nodes if node.parent is None]
        num_roots = len(roots)
        if num_roots == 0:
            raise ValueError("No root nodes found")
        if num_roots > 1:
            raise ValueError("Multiple root nodes found")
        return roots[0]

    def _get_leaves(self) -> list[Node]:
        return [node for node in self.nodes if not node.children]
    
    def _propagate_current_amounts_up(self):
        def _get_amount(node_id: str) -> float:
            children = self._get_children(node_id)
            if not children:
                to_return = self._get_node(node_id).current_amount
            else:
                to_return = sum(
                    [_get_amount(child_id) for child_id in children]
                )
                self._get_node(node_id).current_amount = to_return
            return to_return
        _get_amount(self._get_root().id)

    def _propagate_target_amounts_down(self):
        root_node = self._get_root()
        root_node.target_amount = root_node.current_amount
        current_parents = [root_node.id]
        while current_parents:
            new_parents = []
            for parent_id in current_parents:
                parent_node = self._get_node(parent_id)
                target_amount = parent_node.target_amount
                children = self._get_children(parent_id)
                if not children:
                    continue

                # remove frozen children's target_amount from target_amount
                frozen_children = [child_id for child_id in children if self._get_node(child_id).frozen]
                frozen_amount = sum([self._get_node(child_id).target_amount for child_id in frozen_children])
                target_amount -= frozen_amount

                # distribute remaining target_amount to remaining children
                remaining_children = [child_id for child_id in children if not self._get_node(child_id).frozen]
                combined_weight = sum([self._get_node(child_id).weight for child_id in remaining_children])
                for child_id in remaining_children:
                    child_node = self._get_node(child_id)
                    child_node.target_amount = target_amount * child_node.weight / combined_weight
                    new_parents.append(child_id)
            current_parents = new_parents
        
    def _propagate_frozen_status_up_one_step(self):
        for node in self.nodes:
            children = self._get_children(node.id)
            if children and all([self._get_node(child_id).frozen for child_id in children]):
                node.target_amount = sum([self._get_node(child_id).target_amount for child_id in children])
                node.frozen = True

    def _propagate_frozen_status_up(self):
        num_frozen_nodes_before = 0
        num_frozen_nodes_after = sum([node.frozen for node in self.nodes])
        while num_frozen_nodes_before != num_frozen_nodes_after:
            num_frozen_nodes_before = num_frozen_nodes_after
            self._propagate_frozen_status_up_one_step()
            num_frozen_nodes_after = sum([node.frozen for node in self.nodes]) 

    def _equilibrate(self):
        self._find_children()
        self._propagate_current_amounts_up()
        self._propagate_frozen_status_up()
        self._propagate_target_amounts_down()      
    
    def plot_graph(self, output_path: str = "graph.svg"):
        graph = Digraph()

        total_amount = self._get_root().current_amount
        min_font_size = 8
        max_font_size = 16
        for node in self.nodes:
            color = 'lightblue' if node.frozen else 'orange'
            # Customize node appearance
            graph.node(
                node.id,
                label=node.print_name + f"\n {int(round(node.current_amount,ndigits=0))}",
                shape='box',
                style='filled',
                color=color,
                fontname='Helvetica',
                fontsize='10',
            )
            if node.parent:
                fraction = node.current_amount / total_amount
                curr_font_size = max(min_font_size, max_font_size * np.sqrt(fraction))
                # Customize edge appearance
                graph.edge(
                    node.parent,
                    node.id,
                    color='black',
                    #weight=str(node.target_amount),
                    label=(str(int(round(100*node.weight,ndigits=0)))+"%" if not node.frozen else "fixed") + " = " + str(int(round(node.target_amount,ndigits=0))),
                    fontsize=str(int(curr_font_size)),
                )
        return graph.render(output_path, format="png", cleanup=True)


    def absorb_graph(
        self,
        graph_that_is_absorbed,
        node_id_to_attach_to: str = "",
    ):
        sub_root_id = graph_that_is_absorbed._get_root().id

        self.nodes += graph_that_is_absorbed.nodes
        self._get_node(sub_root_id).parent = node_id_to_attach_to
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
                        "current_amount": round(node.current_amount,2),
                        "target_amount": round(node.target_amount,2),
                    }
                )
    
    @classmethod
    def from_toml(cls, toml_path: str):
        def parse_nodes(data, parent_id=None):
            nodes = []
            for key, val in data.items():
                if isinstance(val, dict):
                    node_id = key.split('.')[-1]
                    cur_node = Node(
                        id=node_id,
                        print_name=val.get("print_name", None),
                        parent=parent_id,
                        current_amount=val.get("current_amount", None),
                        target_amount=val.get("target_amount", None),
                        weight=val.get("weight", 1.0),
                    )
                    nodes.append(cur_node)
                    nodes.extend(parse_nodes(val, parent_id=node_id))
            return nodes

        with open(toml_path, "r") as file:
            data = toml.load(file)

        nodes = parse_nodes(data)
        return cls(nodes=nodes)

if __name__ == "__main__":
    input_path = Path(__file__).parent.parent.parent / "data" / "input" / "graph"
    output_path = Path(__file__).parent.parent.parent / "data" / "output"

    #read graphs
    base_graph = Graph.from_toml(input_path / "main.toml")
    elisa_graph = Graph.from_toml(input_path / "elisa.toml")
    set_aside_graph = Graph.from_toml(input_path / "set_aside.toml")
    investment_graph = Graph.from_toml(input_path / "investments.toml")
    crypto_graph = Graph.from_toml(input_path / "crypto.toml")

    #combine graphs
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

    #plot result
    base_graph.plot_graph(output_path / "graph") # inner function appends .png suffix

    #save graph to csv
    base_graph.save_to_csv(output_path / "graph_history.csv")