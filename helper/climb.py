def climb():
    edges, nodes = generate_full_mesh(50)
    transceivers = [(40, 1), (100, 3), (500, 20), (1000, 50)]
    chrom = random_chromosome(nodes, transceivers)
    new_net = paralelize.Network(90, 30, edges, nodes)
    new_net.set_shortest_paths()
