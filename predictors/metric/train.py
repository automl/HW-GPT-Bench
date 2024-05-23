from predictors.metric.net import Net
from predictors.metric.utils import PPLDataset, search_spaces
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import scipy
def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()
        target = target.float()
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(torch.squeeze(output), torch.squeeze(target))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    out_test = []
    test_target = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.float()
            target = torch.squeeze(target)
            test_target.append(torch.squeeze(target))
            output = model(data)
            output = torch.squeeze(output)
            out_test.append(torch.squeeze(output))
            test_loss += nn.MSELoss()(output, target).item() # sum up batch loss
    print(target[-1])
    print(output[-1])
    final_target = torch.cat(test_target,dim=-1)
    final_out = torch.cat(out_test,dim=-1)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    print("Corr", scipy.stats.kendalltau(final_target.cpu().numpy(), final_out.cpu().numpy())[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch PPL Predictor')
    parser.add_argument('--metric', type=str, default='perplexity',)
    parser.add_argument('--search_space', type=str, default='s',
                        help='search space')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=4000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed ((default: 1)')
    parser.add_argument('--save_path', type=str, default='ppl_predictor_ckpts/',
                        help='path to save the model checkpoints')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataset = PPLDataset(search_space =args.search_space, metric = args.metric)
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    # get the search space
    choices_dict = search_spaces[args.search_space]
    num_layers = max(choices_dict['n_layer_choices'])
    layer_size = 128
    model = Net(num_layers, layer_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    save_path = args.save_path  + args.metric + "_" + args.search_space #+ "_hw_embed_on_" + str(hw_embed_on) + "_hw_embed_dim_" + str(hw_embed_dim) + "_layer_size_" + str(layer_size) + "_epochs_" + str(args.epochs) + "_lr_" + str(args.lr) + "_seed_" + str(args.seed) + ".pt"
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        if epoch%10 == 0:
           test(model, device, test_loader)
        torch.save(model.state_dict(), save_path+".pt")


    