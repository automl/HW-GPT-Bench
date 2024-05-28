from hwgpt.predictors.hwmetric.net import Net
from hwgpt.predictors.hwmetric.utils import HWDataset, search_spaces
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import scipy
from hwgpt.predictors.hwmetric.utils import get_model_and_datasets
import pickle


def train(
    model: torch.nn.Module,
    device: str,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Adam,
    epoch: int,
    log_interval: int = 10,
):
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


def test(model: torch.nn.Module, device: str, test_loader: torch.utils.data.DataLoader):
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
    # print(final_target.shape)
    # print(final_out.shape)
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
        default="energies",
    )
    parser.add_argument("--search_space", type=str, default="s")
    parser.add_argument("--model", type=str, default="conformal_quantile")
    parser.add_argument("--type", type=str, default="quantile")
    parser.add_argument("--num_quantiles", type=str, default=9)
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
        default=4000,
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
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model, train_dataset, test_dataset = get_model_and_datasets(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1024, shuffle=False
    )
    if args.model == "conformal_quantile" or args.model == "quantile":
        X_train = train_dataset.arch_features_train.data.numpy()
        X_test = test_dataset.arch_features_test.data.numpy()
        Y_train = train_dataset.latencies_train.data.numpy()
        Y_test = test_dataset.latencies_test.data.numpy()
        model.fit(X_train, Y_train)
        base_path = (
            "data_collection/gpt_datasets/predictor_ckpts/hwmetric/"
            + str(args.model)
            + "/"
        )
        model_path = (
            base_path
            + args.metric
            + "_"
            + args.type
            + "_"
            + args.search_space
            + "_"
            + args.device
            + ".pkl"
        )
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = model.to(device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_path = (
            "data_collection/gpt_datasets/predictor_ckpts/hwmetric/"
            + str(args.model)
            + "/"
        )
        model_path = (
            base_path
            + args.metric
            + "_"
            + args.type
            + "_"
            + args.search_space
            + "_"
            + args.device
            + ".pth"
        )
        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
        torch.save(model.state_dict(), model_path)
