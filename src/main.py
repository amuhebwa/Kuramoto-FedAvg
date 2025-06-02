import torch
import json
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import algorithms
from train_tools import *
#from utils import *

import numpy as np
import argparse
import warnings
import wandb
import random
import pprint
import os

warnings.filterwarnings("ignore")

# Set torch base print precision
torch.set_printoptions(10)

__all__ = ["ConfLoader", "directory_setter", "config_overwriter"]

class ConfLoader:

    """
    Load json config file using DictWithAttributeAccess object_hook.
    ConfLoader(conf_name).opt attribute is the result of loading json config file.
    """

    class DictWithAttributeAccess(dict):
        """
        This inner class makes dict to be accessed same as class attribute.
        For example, you can use opt.key instead of the opt['key']
        """

        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    def __init__(self, conf_name):
        self.conf_name = conf_name
        self.opt = self.__get_opt()

    def __load_conf(self):
        with open(self.conf_name, "r") as conf:
            opt = json.load(
                conf, object_hook=lambda dict: self.DictWithAttributeAccess(dict)
            )
        return opt

    def __get_opt(self):
        opt = self.__load_conf()
        opt = self.DictWithAttributeAccess(opt)

        return opt

def directory_setter(path="./results", make_dir=False):
    """
    Make dictionary if not exists.
    """
    if not os.path.exists(path) and make_dir:
        os.makedirs(path)  # make dir if not exist
        print("directory %s is created" % path)

    if not os.path.isdir(path):
        raise NotADirectoryError(
            "%s is not valid. set make_dir=True to make dir." % path
        )

def config_overwriter(opt, args):
    """
    Overwrite loaded configuration by parsing arguments.
    """
    if args.dataset_name is not None:
        opt.data_setups.dataset_name = args.dataset_name

    if args.batch_size is not None:
        opt.data_setups.batch_size = args.batch_size

    if args.n_clients is not None:
        opt.data_setups.n_clients = args.n_clients

    if args.partition_method is not None:
        opt.data_setups.partition.method = args.partition_method

    if args.partition_s is not None:
        opt.data_setups.partition.shard_per_user = args.partition_s

    if args.partition_alpha is not None:
        opt.data_setups.partition.alpha = args.partition_alpha

    if args.model_name is not None:
        opt.train_setups.model.name = args.model_name

    if args.n_rounds is not None:
        opt.train_setups.scenario.n_rounds = args.n_rounds

    if args.sample_ratio is not None:
        opt.train_setups.scenario.sample_ratio = args.sample_ratio

    if args.local_epochs is not None:
        opt.train_setups.scenario.local_epochs = args.local_epochs

    if args.device is not None:
        opt.train_setups.scenario.device = args.device

    if args.lr is not None:
        opt.train_setups.optimizer.params.lr = args.lr

    if args.momentum is not None:
        opt.train_setups.optimizer.params.momentum = args.momentum

    if args.wd is not None:
        opt.train_setups.optimizer.params.weight_decay = args.wd

    if args.algo_name is not None:
        opt.train_setups.algo.name = args.algo_name

    if args.seed is not None:
        opt.train_setups.seed = args.seed

    if args.group is not None:
        opt.wandb_setups.group = args.group

    if args.exp_name is not None:
        opt.wandb_setups.name = args.exp_name

    return opt

ALGO = {
    "fedavg": algorithms.fedavg.Server,
    "scaffold": algorithms.scaffold.Server,
    "kuramoto": algorithms.scaffold.Server,
}

SCHEDULER = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
}

def _get_setups(args):
    """Get train configuration"""

    # Fix randomness for data distribution
    np.random.seed(19940817)
    random.seed(19940817)

    # Distribute the data to clients
    data_distributed = data_distributer(**args.data_setups)

    # Fix randomness for experiment
    _random_seeder(args.train_setups.seed)
    model = create_models(
        args.train_setups.model.name,
        args.data_setups.dataset_name,
        **args.train_setups.model.params,
    )

    # Optimization setups
    optimizer = optim.SGD(model.parameters(), **args.train_setups.optimizer.params)
    scheduler = None

    if args.train_setups.scheduler.enabled:
        scheduler = SCHEDULER[args.train_setups.scheduler.name](
            optimizer, **args.train_setups.scheduler.params
        )

    # Algorith-specific global server container
    algo_params = args.train_setups.algo

    server = ALGO[args.train_setups.algo.name](algo_params, model, data_distributed, optimizer, scheduler, **args.train_setups.scenario,)

    return server


def _random_seeder(seed):
    """Fix randomness"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    """Execute experiment"""

    # Load the configuration
    server = _get_setups(args)

    # Conduct FL
    server.run()

    # Save the final global model
    #model_path = os.path.join(wandb.run.dir, "model.pth")
    #torch.save(server.model.state_dict(), model_path)

    # Upload model to wandb
    #wandb.save(model_path)

# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Process Configs")
parser.add_argument("--config_path", default="./config/fedavg.json", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--n_clients", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--partition_method", type=str)
parser.add_argument("--partition_s", type=int)
parser.add_argument("--partition_alpha", type=float)
parser.add_argument("--model_name", type=str)
parser.add_argument("--n_rounds", type=int)
parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--local_epochs", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--momentum", type=float)
parser.add_argument("--wd", type=float)
parser.add_argument("--algo_name", type=str)
parser.add_argument("--device", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--group", type=str)
parser.add_argument("--exp_name", type=str)
args = parser.parse_args()

#######################################################################################

if __name__ == "__main__":
    
    # Load configuration from .json file
    opt = ConfLoader(args.config_path).opt

    # Overwrite config by parsed arguments
    opt = config_overwriter(opt, args)

    # Print configuration dictionary pretty
    print("")
    print("=" * 50 + " Configuration " + "=" * 50)
    pp = pprint.PrettyPrinter(compact=True)
    pp.pprint(opt)
    print("=" * 120)

    # Initialize W&B
    wandb.init(config=opt, **opt.wandb_setups)

    # How many batches to wait before logging training status
    wandb.config.log_interval = 10

    # Execute expreiment
    main(opt)
