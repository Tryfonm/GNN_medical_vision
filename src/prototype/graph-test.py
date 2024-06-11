import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Assuming you have your 3D numpy array with cluster labels
labels_array = np.random.randint(0, 10, size=(10, 10, 8))  # Example random labels, replace it with your actual data

# Create a NetworkX graph
G = nx.Graph()

# Iterate through the array to identify clusters and their connections
for i in range(labels_array.shape[0]):
    for j in range(labels_array.shape[1]):
        for k in range(labels_array.shape[2]):
            current_label = labels_array[i, j, k]
            
            # Check neighboring voxels for the same label to establish connections
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if (dx != 0 or dy != 0 or dz != 0) and \
                           0 <= i + dx < labels_array.shape[0] and \
                           0 <= j + dy < labels_array.shape[1] and \
                           0 <= k + dz < labels_array.shape[2]:
                            
                            neighbor_label = labels_array[i + dx, j + dy, k + dz]
                            
                            if current_label != neighbor_label:
                                # Add an edge between different labels
                                G.add_edge(current_label, neighbor_label)

# Visualize the graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, dim=2)  # You might need to adjust layout algorithms
nx.draw(G, pos, with_labels=True, node_size=100, node_color='skyblue', edge_color='gray')
plt.title('Network Visualization of Clusters')
plt.show()
