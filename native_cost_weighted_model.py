"""
Native White-Box Cost-Weighted Binary Classification Engine for SECOM
Implements:
  1. 14x Defect-Penalized Weighted Cross-Entropy Loss and Analytical Gradients
  2. Balanced Mini-Batch Array/Index Shuffling Loop
  3. Calibrated Lower-Threshold Sigmoid Decision Boundary (Threshold = 0.10 - 0.12)
"""

import numpy as np


class NativeCostWeightedClassifier:
    def __init__(self, learning_rate=0.01, num_epochs=300, batch_size=64,
                 pos_cost_weight=14.07, reg_lambda=0.01, threshold=0.10):
        """
        Custom White-Box Cost-Weighted Logistic Optimization Model.
        
        Parameters:
        -----------
        pos_cost_weight : float, default=14.07
            14x penalty weight applied to minority defect misclassifications (14:1 ratio).
        threshold : float, default=0.10
            Calibrated lower decision threshold on raw sigmoid probability for defect classification.
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.pos_cost_weight = pos_cost_weight  # 14.0x penalty for Defect (1)
        self.neg_cost_weight = 1.0              # 1.0x penalty for Pass (0)
        self.reg_lambda = reg_lambda
        self.threshold = threshold              # Lower decision boundary for severe imbalance
        self.weights = None
        self.bias = 0.0

    @staticmethod
    def sigmoid(z):
        """Numerically stable sigmoid activation."""
        z = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_weighted_loss_and_gradients(self, X_batch, y_batch):
        """
        Computes 14x Defect-Penalized Cost and Gradients mathematically:
        
        Loss:
          L = -1/N * sum( w_+ * y * log(p) + w_- * (1-y) * log(1-p) ) + lambda/2 * ||w||^2
        
        Gradient w.r.t weights:
          dL/dw = 1/N * X^T * [ (w_+ * y + w_- * (1-y)) * p - w_+ * y ] + lambda * w
        """
        N = X_batch.shape[0]
        z = np.dot(X_batch, self.weights) + self.bias
        p = self.sigmoid(z)
        eps = 1e-12

        # 1. Weighted Cost calculation
        w_i = y_batch * self.pos_cost_weight + (1.0 - y_batch) * self.neg_cost_weight
        loss = -np.mean(
            self.pos_cost_weight * y_batch * np.log(p + eps) +
            self.neg_cost_weight * (1.0 - y_batch) * np.log(1.0 - p + eps)
        ) + 0.5 * self.reg_lambda * np.sum(self.weights ** 2)

        # 2. Weighted Gradient calculation
        # Error signal scaled by 14x for positive defect instances
        error = (w_i * p) - (y_batch * self.pos_cost_weight)
        grad_w = (1.0 / N) * np.dot(X_batch.T, error) + self.reg_lambda * self.weights
        grad_b = (1.0 / N) * np.sum(error)

        return loss, grad_w, grad_b

    def fit(self, X, y):
        """
        Train using pure array/index-shuffling balanced batching loop.
        """
        num_samples, num_features = X.shape
        # Initialize weights with Xavier scaling
        np.random.seed(42)
        self.weights = np.random.randn(num_features) * np.sqrt(2.0 / num_features)
        self.bias = 0.0

        # Separate minority defect and majority pass indices for balanced batching
        idx_defect = np.where(y == 1)[0]
        idx_pass = np.where(y == 0)[0]
        half_batch = self.batch_size // 2

        for epoch in range(self.num_epochs):
            # Shuffle indices within each epoch
            np.random.shuffle(idx_defect)
            np.random.shuffle(idx_pass)

            # Balanced batch iteration
            num_batches = max(1, num_samples // self.batch_size)
            epoch_loss = 0.0

            for b in range(num_batches):
                # Resample minority defect indices to maintain balance in every step
                batch_defect_idx = np.random.choice(idx_defect, size=half_batch, replace=True)
                batch_pass_idx = np.random.choice(idx_pass, size=half_batch, replace=False)
                
                batch_indices = np.concatenate([batch_defect_idx, batch_pass_idx])
                np.random.shuffle(batch_indices)

                X_b = X[batch_indices]
                y_b = y[batch_indices]

                loss, grad_w, grad_b = self.compute_weighted_loss_and_gradients(X_b, y_b)
                
                # Parameter update step
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b
                epoch_loss += loss

        return self

    def predict_proba(self, X):
        """Compute raw sigmoid failure probability."""
        z = np.dot(X, self.weights) + self.bias
        p1 = self.sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        """
        Decision boundary calibrated to lower threshold (e.g. 0.10) to catch all defects.
        """
        p_defect = self.predict_proba(X)[:, 1]
        return (p_defect >= self.threshold).astype(int)
