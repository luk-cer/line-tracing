class AttributeProxy:
    """Helper to allow graph.edges.attr_name[indices] syntax."""
    def __init__(self, data_ref, mask_ref, attr_map):
        self._data = data_ref
        self._mask = mask_ref
        self._map = attr_map

    def __getattr__(self, name):
        if name in self._map:
            # Returns a view of the attribute layer
            return self._data[self._map[name]]
        raise AttributeError(f"Attribute '{name}' not found.")

    @property
    def mask(self):
        """Access the global mask directly via .edges.mask or .nodes.mask"""
        return self._mask

class FullyConnectedGraph:
    def __init__(self, num_nodes, edge_attr_names, vert_attr_names):
        self.num_nodes = num_nodes
        
        # Mappings for attribute names to indices
        self.e_map = {name: i for i, name in enumerate(edge_attr_names)}
        self.v_map = {name: i for i, name in enumerate(vert_attr_names)}
        
        # Data storage: Attributes as first dimension for fast layer slicing
        # Shape: (Num_Attrs, Nodes, Nodes)
        self.e_data = np.zeros((len(edge_attr_names), num_nodes, num_nodes))
        self.v_data = np.zeros((len(vert_attr_names), num_nodes))
        
        # Global Boolean Masks
        self.e_mask = np.ones((num_nodes, num_nodes), dtype=bool)
        self.v_mask = np.ones(num_nodes, dtype=bool)
        
        # Proxies for clean access: graph.edges.weight or graph.nodes.temp
        self.edges = AttributeProxy(self.e_data, self.e_mask, self.e_map)
        self.nodes = AttributeProxy(self.v_data, self.v_mask, self.v_map)

    def set_edge(self, u, v, **kwargs):
        """Sets multiple edge attributes and maintains undirected symmetry."""
        for attr, value in kwargs.items():
            idx = self.e_map[attr]
            self.e_data[idx, u, v] = value
            self.e_data[idx, v, u] = value

    def set_edge_mask(self, u_indices, v_indices, status: bool):
        """Bulk update edge masks using fancy indexing."""
        self.e_mask[u_indices, v_indices] = status
        self.e_mask[v_indices, u_indices] = status

    def set_node_mask(self, indices, status: bool):
        """Bulk update node masks."""
        self.v_mask[indices] = status

    def get_active_edge_data(self, attr_name):
        """Returns a flat array of attribute values for all active edges."""
        layer = getattr(self.edges, attr_name)
        return layer[self.e_mask]