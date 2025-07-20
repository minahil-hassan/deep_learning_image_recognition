import os
import itertools
import json

# Create directory to save configs
os.makedirs("experiment_configs", exist_ok=True)
print("Working directory:", os.getcwd())


# Define hyperparameter search space
learning_rates = [0.01, 0.001, 0.0001]
dropouts = [0.2, 0.3]
dense_units_list = [512, 128]
optimisers = ['adam', 'adagrad', 'rmsprop']
activations = ['relu', 'elu']
num_epochs_list = [30, 50]

# Generate all combinations of hyperparameters
experiments = []
for lr, dr, units, opt, act, n_ep in itertools.product(
    learning_rates, dropouts, dense_units_list, optimisers, activations, num_epochs_list
):
    exp_name = f"{opt}_lr{lr}_drop{dr}_units{units}_{act}"
    exp_cfg = {
        "name": exp_name,
        "dense_units": units,
        "dropout_rate": dr,
        "learning_rate": lr,
        "optimiser": opt,
        "activation": act,
        "num_epochs": n_ep,
        "train_entire_model": True
    }
    experiments.append(exp_cfg)

    # Save each config as a JSON file
    with open(f"experiment_configs/{exp_name}.json", "w") as f:
        json.dump(exp_cfg, f, indent=2)

print(f"Generated {len(experiments)} experiment configs in 'experiment_configs/'")
