import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.server import ServerConfig
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
from datetime import datetime
import os

# Set random seed for reproducibility
np.random.seed(42)

# Experiment metadata
EXPERIMENT_ID = f"fed_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(f"results/{EXPERIMENT_ID}", exist_ok=True)

# Metrics storage
round_accuracies = []
round_losses = []
client_accuracy_log = {}
sent_bytes_log = []
client_id_map = {}

class LoggingFedAvg(FedAvg):
    """Custom FedAvg strategy with enhanced logging for research purposes."""
    def aggregate_fit(self, rnd, results, failures):
        """Aggregate client model updates and log communication costs."""
        if failures:
            print(f"[Round {rnd}] Failures detected: {len(failures)} clients failed.")
        total_bytes = 0
        for client_id, res in results:
            sent = res.metrics.get("sent_bytes", 0)
            total_bytes += sent
        sent_bytes_log.append((rnd, total_bytes))
        print(f"[Round {rnd}] Total Data Sent: {total_bytes/(1024**2):.2f} MB")
        return super().aggregate_fit(rnd, results, failures)

    def aggregate_evaluate(self, rnd, results, failures):
        """Aggregate client evaluation metrics and log detailed results."""
        client_accs = {}
        client_losses = {}
        for client_obj, res in results:
            acc = res.metrics.get("accuracy")
            loss = res.metrics.get("loss")
            if acc is not None and loss is not None:
                if client_obj not in client_id_map:
                    client_id_map[client_obj] = f"Client {len(client_id_map)}"
                readable_id = client_id_map[client_obj]
                client_accs[readable_id] = acc
                client_losses[readable_id] = loss
                client_accuracy_log.setdefault(readable_id, []).append(acc)

        if client_accs:
            avg_acc = sum(client_accs.values()) / len(client_accs)
            avg_loss = sum(client_losses.values()) / len(client_losses)
            round_accuracies.append((rnd, avg_acc))
            round_losses.append((rnd, avg_loss))
            print(f"[Round {rnd}] Avg Accuracy: {avg_acc:.4f}, Avg Loss: {avg_loss:.4f}")
            for cid in client_accs:
                print(f" - {cid}: Acc={client_accs[cid]:.4f}, Loss={client_losses[cid]:.4f}")
        else:
            print(f"[Round {rnd}] No evaluation data available.")
        return super().aggregate_evaluate(rnd, results, failures)

def save_metrics():
    """Save experiment metrics to CSV and JSON files."""
    with open(f"results/{EXPERIMENT_ID}/communication_cost.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Total_Sent_Bytes_MB"])
        for rnd, sent in sent_bytes_log:
            writer.writerow([rnd, sent/(1024**2)])

    with open(f"results/{EXPERIMENT_ID}/metrics.json", "w") as f:
        json.dump({
            "round_accuracies": round_accuracies,
            "round_losses": round_losses,
            "client_accuracy_log": client_accuracy_log
        }, f, indent=4)

def plot_metrics():
    """Generate and save plots for accuracy and communication cost."""
    if not round_accuracies:
        print("No metrics to plot.")
        return

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    rounds, accs = zip(*round_accuracies)
    plt.plot(rounds, accs, marker='o', label='Global Accuracy')
    for cid, acc_list in client_accuracy_log.items():
        plt.plot(range(1, len(acc_list)+1), acc_list, label=cid, alpha=0.7)
    plt.title("Federated Learning Accuracy per Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{EXPERIMENT_ID}/accuracy_plot.png", dpi=300)
    plt.close()

    # Communication cost plot
    plt.figure(figsize=(10, 6))
    rounds, bytes_sent = zip(*sent_bytes_log)
    plt.plot(rounds, [b/(1024**2) for b in bytes_sent], marker='o', color='r')
    plt.title("Communication Cost per Round")
    plt.xlabel("Round")
    plt.ylabel("Data Sent (MB)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"results/{EXPERIMENT_ID}/communication_plot.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    strategy = LoggingFedAvg(
        min_fit_clients=config["min_fit_clients"],
        min_evaluate_clients=config["min_evaluate_clients"],
        min_available_clients=config["min_available_clients"],
    )

    try:
        fl.server.start_server(
            server_address=config["server_address"],
            config=ServerConfig(num_rounds=config["num_rounds"]),
            strategy=strategy,
        )
    except Exception as e:
        print(f"Server error: {str(e)}")
    finally:
        save_metrics()
        plot_metrics()