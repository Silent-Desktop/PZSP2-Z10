import numpy as np
from numpy.typing import NDArray
from typing import Optional


class AMPL_Data_Creator:
    def generate_ampl_data(
        self,
        encrypted_bandwidth: int,
        regular_bandwidth: int,
        nodes: int | list[int],
        transponders: int | list[str],
        demand: int | float | NDArray[np.float32],
        trans_capacities: dict[str, int | float] | NDArray[np.float32],
        trans_costs: dict[str, int | float] | NDArray[np.float32],
        edges: Optional[list[tuple[int, int]]] = None,
    ):
        """Generate AMPL .dat file contents."""
        # handle parameter types and values
        if encrypted_bandwidth < 0 or regular_bandwidth < 0:
            raise ValueError("Bandwidth limit cannot be less than zero!")
        int_nodes, int_edges = self._check_nodes_edges(nodes, edges)
        transponders_array = self._transponders_array(transponders)
        try:
            len(demand)
            self._scalar_demand = False
        except TypeError:
            if demand < 0:
                raise ValueError("Demand shouldn't be negative!")
            self._scalar_demand = True
        if not self._scalar_demand:
            n = len(int_nodes)
            try:
                if demand.shape != (n, n):
                    raise ValueError(
                        f"Demand shape {demand.shape} doesn't fit node amount {n}"
                    )
            except AttributeError:
                if len(demand) != n or any([len(row) != n for row in demand]):
                    raise ValueError(
                        f"Demand shape doesn't fit node amount {n}"
                    )
        if edges is None:
            output = self._gen_full_mesh(int_nodes)
        else:
            output = self._gen_any_net(int_nodes, int_edges)
        output.extend(
            self._gen_trans_params(
                transponders_array,
                encrypted_bandwidth,
                regular_bandwidth,
                trans_capacities,
                trans_costs,
            )
        )
        if self._scalar_demand:
            output.extend(self._gen_scalar_demand(demand))
        else:
            output.extend(self._gen_matrix_demand(demand))
        return "\n".join(output)

    def _check_nodes_edges(self, nodes, edges):
        try:
            len(nodes)
            try:
                return np.array(nodes, dtype=int), np.array(edges, dtype=int)
            except ValueError:  # nodes/edges not convertible to int
                return np.arange(len(nodes)), self._convert_edges(nodes, edges)
        except TypeError:
            if int(nodes) == nodes and nodes > 0:
                try:
                    arr = np.arange(nodes)
                    return arr, np.array(edges, dtype=int)
                except ValueError:  # edges not convertible to int
                    return arr, self._convert_edges(nodes, edges)
            else:
                raise TypeError(
                    f"Parameter `nodes` should be either:\n"
                    "- a positive integer for generated values,\n"
                    "- or container for custom network!\n"
                    f"Found type: {type(nodes)}"
                )

    def _convert_edges(self, nodes, edges):
        return [
            (
                list(nodes).index(edges[i][0]),
                list(nodes).index(edges[i][1]),
            )
            for i in range(len(edges))
        ]

    def _transponders_array(self, transponders):
        try:
            len(transponders)
        except TypeError:
            if int(transponders) == transponders and transponders > 0:
                return np.array(np.arange(transponders), dtype=str)
            else:
                raise TypeError(
                    f"Parameter `transponders` should be either:\n"
                    "- a positive integer for generated values,\n"
                    "- or container for custom network!\n"
                    f"Found type: {type(transponders)}"
                )

    def _gen_full_mesh(self, n):
        return [f"param n = {n};"]

    def _gen_any_net(self, nodes: list[int], edges: list[tuple[int, int]]):
        return [
            f"set NODES := {" ".join(np.array(nodes, dtype=str))};",
            f"set EDGES := {str(edges)[1:-1]};",
        ]

    def _gen_scalar_demand(self, demand):
        return [f"param demand = {demand};"]

    def _gen_matrix_demand(self, demand: NDArray):
        idx = np.triu_indices_from(demand, k=1)
        demand_lines = [
            f"{i} {j} {demand[i,j]}" for i, j in zip(idx[0], idx[1])
        ]
        demand_lines[-1] += ";"
        return ["param demand :="] + demand_lines

    def _gen_trans_params(
        self,
        transponders: list[str],
        encrypted_bandwidth: int,
        regular_bandwidth: int,
        trans_capacities: dict[str, int | float] | NDArray[np.float32],
        trans_costs: dict[str, int | float] | NDArray[np.float32],
    ):
        output = [
            f"set TRANSPONDERS := {" ".join(transponders)};",
            f"param encrypted_width = {encrypted_bandwidth};",
            f"param base_width = {regular_bandwidth};",
            "param cap :=",
        ]
        try:
            trans_cap_lines = [
                f"{transponder} {trans_capacities[transponder]}"
                for transponder in transponders
            ]
        except (IndexError, TypeError):
            trans_cap_lines = [
                f"{transponders[i]} {trans_capacities[i]}"
                for i in range(len(transponders))
            ]
        try:
            trans_cost_lines = [
                f"{transponder} {trans_costs[transponder]}"
                for transponder in transponders
            ]
        except (IndexError, TypeError):
            trans_cost_lines = [
                f"{transponders[i]} {trans_costs[i]}"
                for i in range(len(transponders))
            ]
        trans_cap_lines[-1] += ";"
        trans_cost_lines[-1] += ";"
        return output + trans_cap_lines + ["param cost :="] + trans_cost_lines