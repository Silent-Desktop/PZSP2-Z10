import numpy as np


class SNDlib_Parser:
    """Parser for SNDlib's native file format"""
    def __init__(self, filename: str):
        self.filename = filename
        self._fh = open(filename)

    def get_data(self):
        """Reads data from `self.filename`.
        Returns:
            - edge list
            - node attribute dict
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

