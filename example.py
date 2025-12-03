from pathlib import PurePath
from amplpy import AMPL
import networkx as nx
import matplotlib.pyplot as plt
import pandas


ampl = AMPL()
ampl.read(str(PurePath("AMPL/PZSP2.mod")))
ampl.read_data(str(PurePath("AMPL/PZSP2.dat")))
N = ampl.get_parameter("n").value()
ampl.option["solver"] = "cplex"
ampl.solve()
assert ampl.solve_result == "solved"
print(f"Total cost: {ampl.get_objective("total_cost").value()}")
flow = ampl.get_variable("usage").get_values().to_pandas()
print(f"Flow matrix:\n{flow}")
encryption = ampl.get_variable("encrypted").get_values().to_pandas()
print(f"Encrypted edges:\n{encryption}")
# bandwidth = flow.groupby(["index0", "index1"]).sum()
# multi_index = [
#     [i for i in range(1, N) for j in range(i+1, N+1)],
#     [j for i in range(1, N) for j in range(i+1, N+1)]
# ]
# values = [bandwidth.at[(i,j), "usage.val"] + bandwidth.at[(j,i), "usage.val"] for i in range(1, N) for j in range(i+1, N+1)]
# bandwidth = pandas.DataFrame(values, index=multi_index, columns=["bandwidth"])
bandwidth = ampl.get_data("{(i,j) in EDGES} sum {(k,l) in CONNECTIONS, m in TRANSPONDERS} (usage[i,j,k,l,m]+usage[j,i,k,l,m])").to_pandas()
bandwidth.rename(columns={bandwidth.columns[0]: "bandwidth"}, inplace=True)
print(f"Bandwidth usage:\n{bandwidth}")
G = nx.Graph()
G.add_nodes_from(range(1,N+1))
G.add_weighted_edges_from([(i,j,bandwidth.at[(i,j),"bandwidth"]) for i in range(1,N+1) for j in range(1,N+1) if i<j])
nx.set_edge_attributes(G, encryption.to_dict()['encrypted.val'], "encrypted")
print(list(G.edges))
print(nx.get_edge_attributes(G, "encrypted"))
colors = [edge[2] for edge in G.edges.data("encrypted")]
pos = nx.arf_layout(G)
nx.draw(G, pos=pos, with_labels=True, font_weight='bold', edge_color=colors)
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=nx.get_edge_attributes(G, "weight"))
plt.show()