import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import torch
import numpy as np
import torch.nn.functional as F


class globalPhysicsInformedDiffusion:
    def __init__(self, diffusion_coef=0.1, time_steps=5, method='implicit'):
        """
        Physics-informed graph diffusion process
        Args:
            diffusion_coef: Diffusion coefficient, controls information propagation speed
            time_steps: Number of time steps, equivalent to GCN layers but with physical meaning
            method: 'implicit' Euler (stable) or 'explicit' Euler (fast)
        """
        self.diffusion_coef = diffusion_coef
        self.time_steps = time_steps
        self.method = method

    def compute_laplacian(self, adj):
        """Compute normalized graph Laplacian L = I - D^(-1/2) A D^(-1/2)"""
        if sp.issparse(adj):
            adj = adj.tocsr()
        else:
            adj = sp.csr_matrix(adj)

        # Degree matrix
        degrees = np.array(adj.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero

        # Normalization
        D_sqrt = sp.diags(1.0 / np.sqrt(degrees))
        L = sp.eye(adj.shape[0]) - D_sqrt @ adj @ D_sqrt
        return L

    def diffuse_features(self, features, adj):
        """Physics diffusion process - alternative to random walk propagation"""
        L = self.compute_laplacian(adj)

        if torch.is_tensor(features):
            features_np = features.cpu().numpy()
        else:
            features_np = features

        if self.method == 'implicit':
            return self._implicit_euler(features_np, L)
        else:
            return self._explicit_euler(features_np, L)

    def _implicit_euler(self, features, L):
        """Implicit Euler method - numerically stable, suitable for large graphs"""
        I = sp.eye(L.shape[0])
        A_diff = I + self.diffusion_coef * L  # (I + αL)u_{t+1} = u_t

        feature_list = [features]
        for t in range(1, self.time_steps):
            # Solve linear system (I + αL)u_{t+1} = u_t
            u_next = []
            for dim in range(features.shape[1]):
                try:
                    u_dim = splinalg.spsolve(A_diff, feature_list[-1][:, dim])
                    u_next.append(u_dim)
                except:
                    # If solving fails, use features from previous time step
                    u_next.append(feature_list[-1][:, dim])

            feature_list.append(np.column_stack(u_next))

        return [torch.FloatTensor(feat) for feat in feature_list]

    def _explicit_euler(self, features, L):
        """Explicit Euler method - faster computation but requires small step size"""
        feature_list = [features]
        for t in range(1, self.time_steps):
            # u_{t+1} = u_t - αL u_t
            u_next = feature_list[-1] - self.diffusion_coef * (L @ feature_list[-1])
            feature_list.append(u_next)

        return [torch.FloatTensor(feat) for feat in feature_list]

    def adaptive_diffusion(self, features, adj, epsilon=0.02):
        """Simplified adaptive diffusion"""
        L = self.compute_laplacian(adj)

        if torch.is_tensor(features):
            features_np = features.cpu().numpy()
        else:
            features_np = features

        # Calculate steady-state solution
        I = sp.eye(L.shape[0])
        A = I + self.diffusion_coef * L

        u_steady = features_np.copy()
        for t in range(self.time_steps):
            u_next = []
            for dim in range(features_np.shape[1]):
                try:
                    u_dim = splinalg.spsolve(A, u_steady[:, dim])
                    u_next.append(u_dim)
                except:
                    u_next.append(u_steady[:, dim])
            u_steady = np.column_stack(u_next)

        # Adaptive diffusion
        feature_list = [features_np]
        hops = np.zeros(adj.shape[0])
        converged = np.zeros(adj.shape[0], dtype=bool)

        for k in range(1, self.time_steps):
            # One step explicit Euler
            u_k = feature_list[-1] - self.diffusion_coef * (L @ feature_list[-1])
            feature_list.append(u_k)

            # Check convergence
            dist = np.linalg.norm(u_k - u_steady, axis=1)
            newly_converged = (dist < epsilon) & (~converged)

            hops[newly_converged] = k
            converged = converged | newly_converged

        hops[~converged] = self.time_steps - 1

        return torch.Tensor(hops), [torch.FloatTensor(feat) for feat in feature_list]


def globalphysics_informed_aver(hops, adj, feature_list, alpha=0.15):
    """Fixed feature fusion - ensure correct 2D output shape"""
    input_feature = []

    if sp.issparse(adj):
        node_degrees = np.array(adj.sum(axis=1)).flatten()
    else:
        node_degrees = np.array(adj.sum(axis=1))

    # Get correct feature dimension
    feat_dim = feature_list[0].shape[1] if feature_list[0].dim() > 1 else 1
    node_count = adj.shape[0]

    print(f"Node count: {node_count}, Feature dimension: {feat_dim}")

    for i in range(node_count):
        hop = int(hops[i].item())

        if hop == 0:
            # Ensure correct feature shape
            fea = feature_list[0][i]
            if fea.dim() == 0:  # If scalar, convert to vector
                fea = fea.unsqueeze(0)
        else:
            # Adaptive weights based on node degree
            degree = node_degrees[i]
            base_weight = 1.0 / (hop + 1)
            degree_weight = np.log(degree + 1) / 10.0

            fea = torch.zeros(feat_dim)  # Initialize as zero vector with correct dimension
            total_weight = 0

            for j in range(min(hop + 1, len(feature_list))):
                current_feat = feature_list[j][i]

                # Ensure current feature has correct shape
                if current_feat.dim() == 0:  # Scalar
                    current_feat = current_feat.unsqueeze(0)

                # Ensure dimension matches
                if current_feat.shape[0] == feat_dim:
                    weight = base_weight * (1 - j / (hop + 1)) + degree_weight
                    fea += weight * ((1 - alpha) * current_feat + alpha * feature_list[0][i])
                    total_weight += weight

            if total_weight > 0:
                fea = fea / total_weight
            else:
                fea = feature_list[0][i]  # Fallback to original feature

        # Ensure feature vector has correct shape
        if fea.dim() == 0:  # If scalar, convert to vector
            fea = fea.unsqueeze(0)

        input_feature.append(fea.unsqueeze(0))  # Add batch dimension

    # Concatenate all node features
    result = torch.cat(input_feature, dim=0)
    print(f"Fused feature shape: {result.shape}")

    # Ensure shape is [node_count, feature_dimension]
    if result.dim() == 1:
        # If 1D, reshape to 2D
        result = result.unsqueeze(1)

    return result


class localPhysicsDiffusion:
    """Label physics diffusion class - newly added"""

    def __init__(self, diffusion_coef=0.05, time_steps=5, method='implicit', alpha=0.8):
        """
        Physics-informed label diffusion
        Args:
            diffusion_coef: Label diffusion coefficient (usually smaller than feature diffusion)
            time_steps: Label diffusion time steps
            method: Numerical method
            alpha: Label preservation coefficient (1-alpha is diffusion strength)
        """
        self.diffusion_coef = diffusion_coef
        self.time_steps = time_steps
        self.method = method
        self.alpha = alpha

    def compute_normalized_adj(self, adj):
        """Compute normalized adjacency matrix"""
        if sp.issparse(adj):
            adj = adj.tocsr()
        else:
            adj = sp.csr_matrix(adj)

        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        degrees = np.array(adj.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        D_sqrt = sp.diags(1.0 / np.sqrt(degrees))
        adj_normalized = D_sqrt @ adj @ D_sqrt
        return adj_normalized

    def diffuse_labels(self, initial_labels, adj, train_mask=None):
        """
        Physics label diffusion process
        Args:
            initial_labels: Initial predicted label probabilities (n_nodes, n_classes)
            adj: Adjacency matrix
            train_mask: Training set mask, ensures training labels remain unchanged
        """
        if torch.is_tensor(initial_labels):
            labels_np = initial_labels.cpu().numpy()
        else:
            labels_np = initial_labels.copy()

        adj_normalized = self.compute_normalized_adj(adj)

        if self.method == 'explicit':
            return self._explicit_label_diffusion(labels_np, adj_normalized, train_mask)
        else:
            return self._implicit_label_diffusion(labels_np, adj_normalized, train_mask)

    def _explicit_label_diffusion(self, labels, adj_norm, train_mask):
        """Explicit Euler label diffusion"""
        label_list = [labels]

        for t in range(1, self.time_steps):
            # Label diffusion: Y_{t+1} = αY_t + (1-α)A_norm Y_t
            diffused = self.alpha * label_list[-1] + (1 - self.alpha) * (adj_norm @ label_list[-1])

            # Keep training labels unchanged
            if train_mask is not None:
                diffused[train_mask] = labels[train_mask]

            label_list.append(diffused)

        return [torch.FloatTensor(label) for label in label_list]

    def _implicit_label_diffusion(self, labels, adj_norm, train_mask):
        """Implicit Euler label diffusion - more stable"""
        n_nodes = labels.shape[0]
        I = sp.eye(n_nodes)

        # Build implicit system: (I + αL) Y_{t+1} = Y_t, where L = I - A_norm
        L = I - adj_norm
        A_implicit = I + self.diffusion_coef * L

        label_list = [labels]

        for t in range(1, self.time_steps):
            # Solve implicit system
            try:
                next_labels = []
                for c in range(labels.shape[1]):  # Process each class separately
                    b = label_list[-1][:, c]
                    y_next = splinalg.spsolve(A_implicit, b)
                    next_labels.append(y_next)

                diffused = np.column_stack(next_labels)

                # Keep training labels unchanged
                if train_mask is not None:
                    diffused[train_mask] = labels[train_mask]

                label_list.append(diffused)

            except Exception as e:
                print(f"Implicit label diffusion failed: {e}, using explicit method")
                return self._explicit_label_diffusion(labels, adj_norm, train_mask)

        return [torch.FloatTensor(label) for label in label_list]

    def adaptive_label_diffusion(self, initial_labels, adj, train_mask=None, epsilon=0.01):
        """Adaptive label diffusion with convergence checking"""

        if torch.is_tensor(initial_labels):
            labels_np = initial_labels.cpu().numpy()
        else:
            labels_np = initial_labels.copy()

        adj_norm = self.compute_normalized_adj(adj)

        # Calculate steady-state solution (simplified)
        steady_state = labels_np

        label_list = [labels_np]
        hops = torch.zeros(adj.shape[0])
        converged = np.zeros(adj.shape[0], dtype=bool)

        for k in range(1, self.time_steps):
            # One step implicit diffusion
            try:
                next_labels = []
                for c in range(labels_np.shape[1]):
                    b = label_list[-1][:, c]
                    y_next = splinalg.spsolve(
                        sp.eye(adj.shape[0]) + self.diffusion_coef * (sp.eye(adj.shape[0]) - adj_norm),
                        b
                    )
                    next_labels.append(y_next)
                current_labels = np.column_stack(next_labels)
            except:
                # If implicit fails, use explicit
                current_labels = self.alpha * label_list[-1] + (1 - self.alpha) * (adj_norm @ label_list[-1])

            # Keep training labels
            if train_mask is not None:
                current_labels[train_mask] = labels_np[train_mask]

            label_list.append(current_labels)

            # Check convergence: label change less than epsilon
            if k > 0:
                label_change = np.linalg.norm(current_labels - label_list[-2], axis=1)
                newly_converged = (label_change < epsilon) & (~converged)

                hops[newly_converged] = k
                converged = converged | newly_converged

        hops[~converged] = self.time_steps - 1

        return hops, [torch.FloatTensor(label) for label in label_list]


def local_physics_aver(hops, adj, label_list, alpha=0.1):
    """Physics-based label fusion"""
    n_nodes = adj.shape[0]
    n_classes = label_list[0].shape[1]

    final_labels = torch.zeros(n_nodes, n_classes)

    for i in range(n_nodes):
        hop = int(hops[i].item())

        if hop == 0:
            # No diffusion
            final_labels[i] = label_list[0][i]
        else:
            # Weighted average of multiple diffusion steps
            total_weight = 0
            weighted_sum = torch.zeros(n_classes)

            for j in range(min(hop + 1, len(label_list))):
                # Weight decays with diffusion steps
                weight = 1.0 / (j + 1) ** 0.5  # Square root decay
                weighted_sum += weight * label_list[j][i]
                total_weight += weight

            if total_weight > 0:
                # Fusion: current diffusion result + original prediction
                final_labels[i] = (1 - alpha) * (weighted_sum / total_weight) + alpha * label_list[0][i]
            else:
                final_labels[i] = label_list[0][i]

    return final_labels