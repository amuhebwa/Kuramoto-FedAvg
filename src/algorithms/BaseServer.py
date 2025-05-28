import torch
import torch.nn as nn
import numpy as np
import copy
import time
import wandb
import os
import csv

from .measures import *

__all__ = ["BaseServer"]

class BaseServer:
    
    def __init__(self, algo_params, model, data_distributed, optimizer, scheduler, n_rounds=200, sample_ratio=0.1, local_epochs=5, device="cuda:0", ):
        """
        Server class controls the overall experiment.
        """
        self.algo_params = algo_params
        self.num_classes = data_distributed["num_classes"]
        self.model = model
        self.testloader = data_distributed["global"]["test"]
        self.criterion = nn.CrossEntropyLoss()
        self.data_distributed = data_distributed
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sample_ratio = sample_ratio
        self.n_rounds = n_rounds
        self.device = device
        self.n_clients = len(data_distributed["local"].keys())
        self.local_epochs = local_epochs
        self.server_results = {
            "client_history": [],
            "test_accuracy": [],
        }

        self.round = 0

        #if (self.algo_params.sync):
        #    self.local_epochs = self.local_epochs + 1

    def run(self):
        """Run the FL experiment"""
        self._print_start()

        for round_idx in range(self.n_rounds):

            #print("Rounds: ", self.local_epochs)

            # Initial Model Statistics
            if round_idx == 0:
                
                test_acc = evaluate_model(self.model, self.testloader, device=self.device)
                self.server_results["test_accuracy"].append(test_acc)    

            start_time = time.time()

            # Make local sets to distributed to clients
            sampled_clients = self._client_sampling(round_idx)
            self.server_results["client_history"].append(sampled_clients)

            # Client training stage to upload weights & stats
            updated_local_weights, client_sizes, round_results, local_grad_norms = self._clients_training(sampled_clients)

            # Get aggregated weights & update global
            if (self.algo_params.sync):
                ag_weights = self._aggregation_with_kuramoto(updated_local_weights, client_sizes, local_grad_norms, K_t=0.3, max_sync_clip=1e3)
            else:
                ag_weights = self._aggregation(updated_local_weights, client_sizes)

            # Update global weights and evaluate statistics
            self._update_and_evaluate(ag_weights, round_results, round_idx, start_time)

            self.round = self.round + 1

    def _clients_training(self, sampled_clients):
        
        """Conduct local training and get trained local models' weights"""
        updated_local_weights, client_sizes = [], []
        round_results = {}
        local_grad_norms = []

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:

            # Fetch client datasets
            self._set_client_data(client_idx)

            # Download global
            self.client.download_global(server_weights, server_optimizer)

            # Local training
            local_results, local_size, avg_grad_norm = self.client.train()
            local_grad_norms.append(avg_grad_norm)

            # Upload locals
            updated_local_weights.append(self.client.upload_local())

            # Update results
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            # Reset local model
            self.client.reset()

        return updated_local_weights, client_sizes, round_results, local_grad_norms

    def _client_sampling(self, round_idx):
        """Sample clients by given sampling ratio"""

        # make sure for same client sampling for fair comparison
        np.random.seed(round_idx)
        clients_per_round = max(int(self.n_clients * self.sample_ratio), 1)
        sampled_clients = np.random.choice(
            self.n_clients, clients_per_round, replace=False
        )

        return sampled_clients

    # def _personalized_evaluation(self):
    #    """Personalized FL performance evaluation for all clients."""

    #     finetune_results = {}

    #     server_weights = self.model.state_dict()
    #     server_optimizer = self.optimizer.state_dict()

    #     # Client finetuning stage
    #     for client_idx in [client_idx for client_idx in self.n_clients]:
    #         self._set_client_data(client_idx)

    #         # Local finetuning
    #         local_results = self.client.finetune(server_weights, server_optimizer)
    #         finetune_results = self._results_updater(finetune_results, local_results)

    #         # Reset local model
    #         self.client.reset()

    #     # Get overall statistics
    #     local_results = {
    #         "local_train_acc": np.mean(round_results["train_acc"]),
    #         "local_test_acc": np.mean(round_results["test_acc"]),
    #     }
    #     wandb.log(local_results, step=round_idx)

    #     return finetune_results

    def _set_client_data(self, client_idx):
        
        """Assign local client datasets."""
        self.client.datasize = self.data_distributed["local"][client_idx]["datasize"]
        self.client.trainloader = self.data_distributed["local"][client_idx]["train"]
        self.client.testloader = self.data_distributed["global"]["test"]

    def _aggregation(self, w, ns):
        
        """Average locally trained model parameters"""

        prop = torch.tensor(ns, dtype=torch.float)
        prop /= torch.sum(prop)
        w_avg = copy.deepcopy(w[0])
        
        for k in w_avg.keys():
            
            w_avg[k] = w_avg[k] * prop[0]

        for k in w_avg.keys():
            
            for i in range(1, len(w)):
                
                w_avg[k] += w[i][k] * prop[i]

        return copy.deepcopy(w_avg)
    
    def _aggregation_with_kuramoto(self, w, ns, local_phases, K_t=0.1, max_sync_clip=1e3):

        """
        Kuramoto-based aggregation with phase synchronization.

        Args:
            w (List[Dict]): List of client model states (each state_dict).
            ns (List[int]): Number of samples per client.
            local_phases (List[float]): Phase angles per client (same length as w).
            K_t (float): Synchronization strength.
            max_sync_clip (float): Optional gradient clipping to stabilize sync correction.

        Returns:
            Dict: Aggregated global weights with synchronization adjustment.
        """

        #print("Kuramoto")

        num_clients = len(w)
        prop = torch.tensor(ns, dtype=torch.float32)
        prop /= torch.sum(prop)

        w_avg = copy.deepcopy(w[0])
        device = next(iter(w_avg.values())).device  # Ensure device compatibility

        for k in w_avg.keys():
            
            if w_avg[k].dtype != torch.float32:
                
                continue  # Skip batch norm stats or non-trainable

            # Start with weighted average initialization
            w_avg[k] = w[0][k].to(device) * prop[0]

            # Add contributions from remaining clients
            for i in range(1, num_clients):
                w_avg[k] += w[i][k].to(device) * prop[i]

            # Add synchronization correction
            sync_term = torch.zeros_like(w_avg[k], device=device)

            for i in range(num_clients):
                for j in range(num_clients):
                    if i == j:
                        continue
                    
                    phase_diff = np.cos(local_phases[j] - local_phases[i])
                    delta_j = w[j][k].to(device) - w_avg[k]
                    sync_term += phase_diff * delta_j

            # Normalize and clip
            sync_term = sync_term / max(1, num_clients - 1)
            sync_term = torch.clamp(sync_term, min=-max_sync_clip, max=max_sync_clip)

            # Add Kuramoto correction
            w_avg[k] += K_t * sync_term

        return copy.deepcopy(w_avg)

    def _results_updater(self, round_results, local_results):
        """Combine local results as clean format"""

        for key, item in local_results.items():
            if key not in round_results.keys():
                round_results[key] = [item]
            else:
                round_results[key].append(item)

        return round_results

    def _print_start(self):
        """Print initial log for experiment"""

        if self.device == "cpu":
            return "cpu"

        if isinstance(self.device, str):
            device_idx = int(self.device[-1])
        elif isinstance(self.device, torch._device):
            device_idx = self.device.index

        device_name = torch.cuda.get_device_name(device_idx)
        print("")
        print("=" * 50)
        print("Train start on device: {}".format(device_name))
        print("=" * 50)

    def _print_stats(self, round_results, test_accs, round_idx, round_elapse):
        
        print(
            "[Round {}/{}] Elapsed {}s (Current Time: {})".format(
                round_idx + 1,
                self.n_rounds,
                round(round_elapse, 1),
                time.strftime("%H:%M:%S"),
            )
        )
        print(
            "[Local Stat (Train Acc)]: {}, Avg - {:2.2f} (std {:2.2f})".format(
                round_results["train_acc"],
                np.mean(round_results["train_acc"]),
                np.std(round_results["train_acc"]),
            )
        )

        #self.update_csv(np.std(round_results["train_acc"]) * 100)

        print(
            "[Local Stat (Test Acc)]: {}, Avg - {:2.2f} (std {:2.2f})".format(
                round_results["test_acc"],
                np.mean(round_results["test_acc"]),
                np.std(round_results["test_acc"]),
            )
        )

        print("[Server Stat] Acc - {:2.2f}".format(test_accs))

        self.update_csv( test_accs * 100, np.std(round_results["train_acc"]))

    def update_csv(self, acc, var):

        filename = f"/home/{os.getenv('USER')}/Kuramoto-FedAvg/evaluation/{self.algo_params.name}/{self.algo_params.data_params}/accuracy.csv"
        
        # Check if the file exists, create it if it doesn't 
        if not os.path.exists(filename):
            
            with open(filename, 'w', newline='') as file:
                
                writer = csv.DictWriter(file, fieldnames=['Accuracy', 'Variance'])
                writer.writeheader()

        # Append the new data to the file
        with open(filename, 'a', newline='') as file:

            writer = csv.DictWriter(file, fieldnames=['Accuracy', 'Variance'])

            # Write a new row with test_loss and test_acc
            writer.writerow({'Accuracy': acc, 'Variance': var})
            
    def _wandb_logging(self, round_results, round_idx):
        """Log on the W&B server"""

        # Local round results
        local_results = {
            "local_train_acc": np.mean(round_results["train_acc"]),
            "local_test_acc": np.mean(round_results["test_acc"]),
        }
        wandb.log(local_results, step=round_idx)

        # Server round results
        server_results = {"server_test_acc": self.server_results["test_accuracy"][-1]}
        wandb.log(server_results, step=round_idx)

    def _update_and_evaluate(self, ag_weights, round_results, round_idx, start_time):
        """Evaluate experiment statistics."""

        # Update Global Server Model
        self.model.load_state_dict(ag_weights)

        # Measure Accuracy Statistics
        test_acc = evaluate_model(self.model, self.testloader, device=self.device,)
        self.server_results["test_accuracy"].append(test_acc)

        # Evaluate Personalized FL performance
        eval_results = get_round_personalized_acc(
            round_results, self.server_results, self.data_distributed
        )
        wandb.log(eval_results, step=round_idx)

        # Change learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        round_elapse = time.time() - start_time

        # Log and Print
        self._wandb_logging(round_results, round_idx)
        self._print_stats(round_results, test_acc, round_idx, round_elapse)
        print("-" * 50)
