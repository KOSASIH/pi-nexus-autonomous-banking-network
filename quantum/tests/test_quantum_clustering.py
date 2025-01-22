import unittest
import numpy as np
from quantum.algorithms.quantum_clustering import QuantumKMeans

class TestQuantumKMeans(unittest.TestCase):
    def setUp(self):
        """Set up test data for the Quantum K-Means algorithm."""
        self.data = np.array([
            [0.1, 0.2],
            [0.2, 0.1],
            [0.9, 0.8],
            [0.8, 0.9],
            [0.5, 0.5]
        ])
        self.kmeans = QuantumKMeans(n_clusters=2)

    def test_fit(self):
        """Test the fit method of Quantum K-Means."""
        self.kmeans.fit(self.data)
        self.assertEqual(len(self.kmeans.centroids), 2, "Centroids should be equal to n_clusters.")
        self.assertTrue(np.all(np.isfinite(self.kmeans.centroids)), "Centroids should be finite numbers.")

    def test_predict(self):
        """Test the predict method of Quantum K-Means."""
        self.kmeans.fit(self.data)
        labels = self.kmeans.predict(self.data)
        self.assertEqual(len(labels), len(self.data), "Labels length should match the data length.")
        self.assertTrue(np.all(np.isin(labels, [0, 1])), "Labels should be in the range of cluster indices.")

    def test_convergence(self):
        """Test the convergence of the Quantum K-Means algorithm."""
        initial_centroids = self.kmeans.centroids.copy()
        self.kmeans.fit(self.data)
        final_centroids = self.kmeans.centroids
        self.assertFalse(np.array_equal(initial_centroids, final_centroids), "Centroids should change after fitting.")

    def test_empty_data(self):
        """Test fitting with empty data."""
        with self.assertRaises(ValueError):
            self.kmeans.fit(np.array([]))

    def test_invalid_n_clusters(self):
        """Test fitting with invalid number of clusters."""
        with self.assertRaises(ValueError):
            QuantumKMeans(n_clusters=0)

if __name__ == "__main__":
    unittest.main()
