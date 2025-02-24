import numpy as np
from sklearn.datasets import load_iris, make_moons

def generate_spiral_dataset(N=8000):
    """Generate a synthetic spiral dataset."""
    pi = np.pi
    theta = np.sqrt(np.random.rand(N)) * 2 * pi
    r_a = 2 * theta + pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(N, 2)
    
    r_b = -2 * theta - pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(N, 2)
    
    x_a = x_a / 20.
    x_b = x_b / 20.
    
    x = np.vstack((x_a, x_b))
    x = (x - x.min()) / (x.max() - x.min())
    y = np.vstack((np.zeros((N, 1)), np.ones((N, 1))))
    
    data = np.hstack((x, y))
    np.random.shuffle(data)
    
    # Use half for training, half for testing
    train_data = data[:N // 2]
    test_data = data[N // 2:]
    X_train, Y_train = train_data[:, :2], train_data[:, 2]
    X_test, Y_test = test_data[:, :2], test_data[:, 2]
    return X_train, Y_train, X_test, Y_test

def load_iris_dataset():
    """Load and normalize the Iris dataset."""
    iris = load_iris()
    data = iris.data
    # Normalize features to [0,1]
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data_norm = (data - data_min) / (data_max - data_min)
    return data_norm, iris.target

def generate_moons_dataset(n_samples=10000, noise=0.1):
    """Generate a Two Moons dataset and normalize it to [0,1]^2."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min)
    return X_norm, y
