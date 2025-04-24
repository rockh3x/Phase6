import numpy as np
from tensorflow.keras.datasets import cifar10

# Define drone-related classes (subset of CIFAR-10)
DRONE_CLASSES = {0: "airplane", 1: "bird", 2: "drone", 3: "helicopter", 4: "kite"}

def create_drone_non_iid(num_clients=5, seed=42):
    """Create non-IID data partitions simulating drone-related classes.
    
    Args:
        num_clients (int): Number of clients to create partitions for.
        seed (int): Random seed for reproducibility.
    
    Returns:
        list: List of (x, y) tuples for each client.
        int: Number of classes (5).
    """
    np.random.seed(seed)
    (x_train, y_train), _ = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    y_train = y_train.flatten()

    # Filter to drone-related classes (assume 0: airplane, 1: bird for simplicity)
    valid_classes = [0, 1]  # Subset for demonstration
    idx = np.where(np.isin(y_train, valid_classes))[0]
    x_train = x_train[idx]
    y_train = y_train[idx]

    # Map to new labels (0, 1)
    label_map = {0: 0, 1: 1}
    y_train = np.array([label_map[y] for y in y_train])

    partitions = []
    sizes = [6000, 3000, 1000, 5000, 4000]
    for i in range(num_clients):
        labels = [(2 * i) % len(valid_classes), (2 * i + 1) % len(valid_classes)]
        idx = np.where(np.isin(y_train, labels))[0]
        np.random.shuffle(idx)
        size = sizes[i % len(sizes)]
        idx = idx[:size]
        partitions.append((x_train[idx], y_train[idx]))
    
    return partitions, len(valid_classes)