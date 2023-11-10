import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # If only one class in the node or max depth reached, create a leaf node
        if len(unique_classes) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return {'class': unique_classes[0], 'count': len(y)}

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            return {'class': np.argmax(np.bincount(y)), 'count': len(y)}

        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Recursive call for left and right child nodes
        left_tree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return {'feature': best_feature, 'threshold': best_threshold,
                'left': left_tree, 'right': right_tree}

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        if num_samples <= 1:
            return None, None

        # Calculate impurity before the split
        initial_impurity = self._calculate_impurity(y)

        best_feature = None
        best_threshold = None
        best_impurity_reduction = 0.0

        # Iterate over each feature and its unique values as potential thresholds
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                # Split the data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                # Skip if any of the subsets is empty
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate impurity after the split
                impurity_left = self._calculate_impurity(y[left_mask])
                impurity_right = self._calculate_impurity(y[right_mask])

                # Calculate impurity reduction
                impurity_reduction = initial_impurity - (
                        (np.sum(left_mask) / num_samples) * impurity_left +
                        (np.sum(right_mask) / num_samples) * impurity_right)

                # Update best split if impurity reduction is higher
                if impurity_reduction > best_impurity_reduction:
                    best_feature = feature
                    best_threshold = threshold
                    best_impurity_reduction = impurity_reduction

        return best_feature, best_threshold

    def _calculate_impurity(self, y):
        # Gini impurity for binary classification
        if len(y) == 0:
            return 0.0
        p_0 = np.sum(y == 0) / len(y)
        p_1 = 1 - p_0
        impurity = 1 - p_0 ** 2 - p_1 ** 2
        return impurity

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree):
        if 'class' in tree:
            return tree['class']
        else:
            if x[tree['feature']] <= tree['threshold']:
                return self._predict_tree(x, tree['left'])
            else:
                return self._predict_tree(x, tree['right'])


