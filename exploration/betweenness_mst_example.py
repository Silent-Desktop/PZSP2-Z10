import networkx as nx
import matplotlib.pyplot as plt

# edges = [(1,2),(1,3),(1,4),(1,5),(1,6),
#          (2,4),(2,6),
#          (3,4),(3,5),
#          (4,6)]
# G = nx.Graph(incoming_graph_data=edges)
G = nx.lollipop_graph(3, 6)
G.add_edges_from([(2,4),(4,6),(6,8)])
betweenness = nx.edge_betweenness_centrality(G, normalized=False)
nx.set_edge_attributes(G, betweenness, "betweenness")
colors = [edge[2] for edge in G.edges.data("betweenness")]

subax1 = plt.subplot(121)
subax1.set_title("Network edge betweenness centrality")
pos = nx.shell_layout(G)
nx.draw(G, pos=pos, with_labels=True, font_weight='bold', edge_color=colors)
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=nx.get_edge_attributes(G, "betweenness"))
subax2 = plt.subplot(122)
subax2.set_title("MST")
nx.draw(nx.minimum_spanning_tree(G, "betweenness"), pos=pos, with_labels=True, font_weight='bold')
plt.show()