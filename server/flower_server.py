import flwr as fl

def main():
    # Define strategy with minimum client requirements
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=3,  # Ensure at least 5 clients are available
        min_fit_clients=3,        # Ensure all 5 clients participate in training
        min_evaluate_clients=3,   # Ensure all 5 clients participate in evaluation
    )

    # Start Flower server
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
