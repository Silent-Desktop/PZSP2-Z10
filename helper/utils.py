def generate_full_mesh(nodes_amount: int):
    edges = []
    for i in range(nodes_amount):
        for j in range(i + 1, nodes_amount):
            edges.append((i, j))
    print(edges)
    return (edges, range(nodes_amount))


# generate_full_mesh(20)
