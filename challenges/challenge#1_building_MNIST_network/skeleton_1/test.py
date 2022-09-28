#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.benchmark as benchmark


#the CNN, RENAME UNIQUELY
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, kernel_size=5)
        self.conv2 = nn.Conv2d(100, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
#training and eval functions

#train function
def train(num_epochs, model, optimizer, train_loader):
    model.train()
    for epoch in range (1, num_epochs+1):    
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_frequency == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] \
                    ({100. * batch_idx / len(train_loader)}%)\tLoss: {loss.item()}')
    #torch.save(model.state_dict(), 'models/model_{num_epochs}.pth')

#val function
def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for image, target in test_loader:
            print(image.shape)
            output = model(image)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
    
    
    print(f'Validation accuracy: {correct}/{len(test_loader.dataset)} \
          ({100. * correct / len(test_loader.dataset)}%)')
    
				
#parameters
n_epochs = 1
batch_size_train = 32
batch_size_test = 256
learning_rate = 0.001
momentum = 0.5
log_frequency = 1000
model_weights = None


#dataloaders
train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data/', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])),
	batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data/', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])),
	batch_size=batch_size_test, shuffle=True)			
			
model = CustomNet()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                        momentum=momentum)

# train(n_epochs, model, optimizer, train_loader)
# test(model, test_loader)


import numpy as np
import time
dummy_input = torch.randn(32, 1, 28, 28) # (B, C, H, W)
# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        t0 = time.time()
        _ = model(dummy_input)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        #ender.record()
        # WAIT FOR GPU SYNC
        #torch.cuda.synchronize()
        #curr_time = starter.elapsed_time(ender)
        
        timings[rep] = t1-t0
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(mean_syn)