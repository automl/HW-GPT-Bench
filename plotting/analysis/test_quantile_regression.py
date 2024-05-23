import torch
import numpy as np 
def get_random_batch(batch_size=32):
   x = np.random.normal(0,1,(batch_size,))
   
   s = np.random.standard_t(df = 3, size=batch_size)


   y = x/3*s*4 + np.sin(x/3*13)*4 + 10
   x = x.reshape(-1,1)
   #y = 4*(x-2)**2 + 3*(x-5)**3 #+ 10*torch.randn_like(x)
   x = torch.tensor(x).float()
   y = torch.tensor(y).float()
   y = torch.squeeze(y)
   # add heteroskedastic noise
   # concat y1,y2,y3,y4,y5,y6,y7,y8,y9,y10
   q = y
   return x, q

# Define the loss function
def quantile_loss(y_pred, y, quantiles):
    quantile_losses = []
    for i, q in enumerate(quantiles):
        errors = y - y_pred[:, i]
        errors_q = torch.max(q*errors, (q-1)*errors)
        quantile_losses.append(torch.mean(errors_q))
    return torch.mean(torch.stack(quantile_losses))

class Net(torch.nn.Module):
    def __init__(self, num_quantiles=4):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.fc3 = torch.nn.Linear(100, num_quantiles)
        self.num_quantiles = num_quantiles

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
quantiles = [0.25, 0.5, 0.75, 0.99]
for epoch in range(20000):
    optimizer.zero_grad()
    x, q = get_random_batch()
    x = x.cuda()
    q = q.cuda()
    y_pred = net(x)
    #y = y.unsqueeze(-1)
    print(q.shape)
    loss = quantile_loss(y_pred, q, quantiles)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, loss {loss.item()}')

#plot the quantiles
import matplotlib.pyplot as plt
x, q = get_random_batch(1024)
x = x.cuda()
q = q.cuda()
plt.scatter(x.cpu().numpy(), q.cpu().numpy(), label='f', color='r', s = 1)
plt.scatter(x.cpu().numpy(), net(x).detach().cpu().numpy()[:,0], label='q1_pred', color='r', marker='x', s = 1)
plt.scatter(x.cpu().numpy(), net(x).detach().cpu().numpy()[:,1], label='q2_pred', color='g', marker='x', s = 1)
plt.scatter(x.cpu().numpy(), net(x).detach().cpu().numpy()[:,2], label='q3_pred', color='b', marker='x', s = 1)
plt.scatter(x.cpu().numpy(), net(x).detach().cpu().numpy()[:,3], label='q4_pred', color='y', marker='x', s = 1)
plt.legend()
plt.savefig("quantile_regression.pdf")