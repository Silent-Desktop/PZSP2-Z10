import numpy as np


class SNDlib_Parser:
    """Parser for SNDlib's native file format"""

    def __init__(self, filename: str):
        self.filename = filename
        self._fh = open(filename)

    def get_nodes_edges_as_ints(self):
        if not self._fh.closed:
            self.get_data()
        try:
            len(self.node_ids)
            try:
                return np.array(self.node_ids, dtype=int), np.array(self.edges, dtype=int)
            except ValueError:  # nodes/edges not convertible to int
                return np.arange(len(self.node_ids)), self._convert_edges()
        except TypeError:
            if int(self.node_ids) == self.node_ids and self.node_ids > 0:
                try:
                    arr = np.arange(self.node_ids)
                    return arr, np.array(self.edges, dtype=int)
                except ValueError:  # edges not convertible to int
                    return arr, self._convert_edges()
            else:
                raise TypeError(
                    f"Parameter `nodes` should be either:\n"
                    "- a positive integer for generated values,\n"
                    "- or container for custom network!\n"
                    f"Found type: {type(self.node_ids)}"
                )
    
    def _convert_edges(self):
        return [
            (
                list(self.node_ids).index(self.edges[i][0]),
                list(self.node_ids).index(self.edges[i][1]),
            )
            for i in range(len(self.edges))
        ]

    def get_data(self):
        """Reads data from `self.filename`.
        Returns:
            - edge list
            - node attribute dict (coordinates for networkx pos)
            - node id list
            - demand matrix (numpy array)"""
        if self._fh.closed:
            self._fh = open(self.filename)
        for line in self._fh:
            if line[0] == "#":
                continue
            line = line.split()
            if not line:
                continue
            match line[0]:
                case "NODES":
                    self._read_nodes()
                case "LINKS":
                    self._read_edges()
                case "DEMANDS":
                    self._read_demand()
                case _:
                    continue
        self._fh.close()
        return self.edges, self.node_attrs, self.node_ids, self.demand

    def _read_nodes(self):
        self._node_attrs = dict()
        for line in self._fh:
            if line[0] == "#":
                continue
            line = line.split()
            if not line:
                continue
            if line[0] == ")":
                break
            self._node_attrs[line[0]] = {
                "pos": (float(line[2]), float(line[3]))
            }
        self._node_ids = list(self._node_attrs.keys())

    @property
    def node_attrs(self) -> dict[str, dict[str, tuple[float, float]]]:
        return self._node_attrs

    @property
    def node_ids(self) -> list[str]:
        return self._node_ids

    def _read_edges(self):
        self._edges = []
        for line in self._fh:
            if line[0] == "#":
                continue
            line = line.split()
            if not line:
                continue
            if line[0] == ")":
                break
            self._edges.append((line[2], line[3]))

    @property
    def edges(self) -> list[tuple[str, str]]:
        return self._edges

    def _read_demand(self):
        N = len(self._node_ids)
        self._demand = np.zeros((N, N), dtype=float)
        for line in self._fh:
            if line[0] == "#":
                continue
            line = line.split()
            if not line:
                continue
            if line[0] == ")":
                break
            i, j = self._node_ids.index(line[2]), self._node_ids.index(line[3])
            self._demand[i, j] = float(line[6])
        self._demand += np.tril(self._demand, k=-1).T
        self._demand = np.triu(self._demand, k=1)

    @property
    def demand(self) -> np.ndarray:
        return self._demand

    def __del__(self):
        self._fh.close()

print(SNDlib_Parser("polska.txt").get_nodes_edges_as_ints())