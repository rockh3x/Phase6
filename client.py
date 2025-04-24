import flwr as fl
import sys
import pickle
import numpy as np
from utils_drone import create_drone_non_iid, DRONE_CLASSES
from model import build_model
from tensorflow.keras.datasets import cifar10

def get_model_size(weights):
    """Calculate size of model weights in bytes."""
    return len(pickle.dumps(weights))

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, x_train, y_train, x_test, y_test, model):
        """Initialize Flower client with data and model."""
        self.client_id = client_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model

    def get_parameters(self, config):
        """Return current model weights."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train model on local data."""
        try:
            self.model.set_weights(parameters)
            history = self.model.fit(
                self.x_train, self.y_train,
                epochs=1, batch_size=32, verbose=0
            )
            new_weights = self.model.get_weights()
            sent_bytes = get_model_size(new_weights)
            print(f"📤 Client {self.client_id} sent ~{sent_bytes/(1024**2):.2f} MB")
            return new_weights, len(self.x_train), {
                "sent_bytes": sent_bytes,
                "train_accuracy": float(history.history["accuracy"][-1])
            }
        except Exception as e:
            print(f"Client {self.client_id} fit error: {str(e)}")
            raise

    def evaluate(self, parameters, config):
        """Evaluate model on local test data."""
        try:
            self.model.set_weights(parameters)
            loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            return loss, len(self.x_test), {
                "accuracy": float(accuracy),
                "loss": float(loss)
            }
        except Exception as e:
            print(f"Client {self.client_id} evaluate error: {str(e)}")
            raise

if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)

    # Load configuration
    import json
    with open("config.json", "r") as f:
        config = json.load(f)

    # Get client ID
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Load data
    partitions, num_classes = create_drone_non_iid(num_clients=config["num_clients"])
    x_train, y_train = partitions[client_id]

    # Load and filter test data
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    y_test = y_test.flatten()
    test_idx = [i for i, y in enumerate(y_test) if y in [0, 1]]  # Match utils_drone.py
    x_test = x_test[test_idx]
    y_test = np.array([0 if y == 0 else 1 for y in y_test[test_idx]])

    # Build model
    model = build_model(input_shape=(32, 32, 3), num_classes=num_classes)

    # Start client
    try:
        fl.client.start_numpy_client(
            server_address=config["server_address"],
            client=FlowerClient(client_id, x_train, y_train, x_test, y_test, model)
        )
    except Exception as e:
        print(f"Client {client_id} startup error: {str(e)}")