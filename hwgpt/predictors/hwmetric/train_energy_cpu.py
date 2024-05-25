from predictors.hwmetric.net import Net
from predictors.hwmetric.utils import HWDataset, search_spaces
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import scipy


def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()
        # print(data.shape)
        # print(target.shape)
        target = target.float()
        optimizer.zero_grad()
        output = model(data)
        target = torch.squeeze(target)
        output = torch.squeeze(output)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    out_test = []
    test_target = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, target = data.float(), target.float()
            target = torch.squeeze(target)
            test_target.append(torch.squeeze(target))
            output = model(data)
            output = torch.squeeze(output)
            out_test.append(torch.squeeze(output))
            test_loss += nn.MSELoss()(output, target).item()  # sum up batch loss
    final_target = torch.cat(test_target, dim=-1)
    final_out = torch.cat(out_test, dim=-1)
    test_loss /= len(test_loader.dataset)
    print(final_target.shape)
    print(final_out.shape)
    print("\nTest set: Average loss: {:.4f}\n".format(test_loss))
    print(
        "Corr",
        scipy.stats.kendalltau(final_target.cpu().numpy(), final_out.cpu().numpy())[0],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch HW Metric Predictor")
    parser.add_argument("--device", type=str, default="a100", help="device name")
    parser.add_argument(
        "--metric",
        type=str,
        default="energy_cpu",
    )
    parser.add_argument("--search_space", type=str, default="", help="search space")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed ((default: 1)"
    )
    parser.add_argument(
        "--hw_embed_on",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="hwmetric_predictor_ckpts/",
        help="path to save the model checkpoints",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    cpus = [
        "cpu_mlgpu",
        "cpu_alldlc",
        "cpu_p100",
        "cpu_p100",
        "cpu_a6000",
        "cpu_meta",
        "helix_cpu",
    ]
    # devices_all = ["P100","a6000", "rtx2080", "rtx3080", "v100", "a100", "a40", "h100", "cpu_mlgpu", "cpu_alldlc", "cpu_p100", "cpu_p100", "cpu_a6000", "cpu_meta", "helix_cpu"]
    models = ["", "m", "l"]
    # gpus = ["a100","a6000", "rtx2080", "rtx3080", "v100", "P100", "a40", "h100"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for device_gpu in cpus:
        for model in models:
            args.search_space = model
            args.device = device_gpu
            kwargs = (
                {"num_workers": 1, "pin_memory": True}
                if torch.cuda.is_available()
                else {}
            )
            train_dataset = HWDataset(
                mode="train",
                device_name=args.device,
                search_space=args.search_space,
                transform=False,
                metric=args.metric,
            )
            test_dataset = HWDataset(
                mode="test",
                device_name=args.device,
                search_space=args.search_space,
                transform=False,
                metric=args.metric,
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=1024, shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1024, shuffle=False
            )
            # get the search space

            if args.search_space == "":
                ss = "s"
            else:
                ss = args.search_space
            choices_dict = search_spaces[ss]
            print(choices_dict)
            num_layers = max(choices_dict["n_layer_choices"])
            hw_embed_on = args.hw_embed_on
            hw_embed_dim = 256
            layer_size = 256
            print(6 + 6 * num_layers + 2)
            model = Net(num_layers, hw_embed_on, hw_embed_dim, layer_size).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            save_path = (
                args.save_path
                + args.device
                + "_"
                + args.metric
                + "_"
                + args.search_space
            )  # + "_hw_embed_on_" + str(hw_embed_on) + "_hw_embed_dim_" + str(hw_embed_dim) + "_layer_size_" + str(layer_size) + "_epochs_" + str(args.epochs) + "_lr_" + str(args.lr) + "_seed_" + str(args.seed) + ".pt"
            for epoch in range(1, args.epochs + 1):
                train(model, device, train_loader, optimizer, epoch)
                test(model, device, test_loader)
                torch.save(model.state_dict(), save_path + ".pt")
