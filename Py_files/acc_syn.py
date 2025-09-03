import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from task import Net, Net_Cifar, GeneratedDataset, F2U_GAN, F2U_GAN_CIFAR
import argparse
import json

parser = argparse.ArgumentParser(description='Train and evaluate model on generated data')

parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
parser.add_argument('--num_chunks_list', nargs="+", type=int, default=[1, 10, 50, 100, 500, 1000, 5000])
parser.add_argument('--epoch_list', nargs="+", type=int, default=[10, 50, 100])
parser.add_argument('--num_samples', type=int, default=10000)

args = parser.parse_args()

dataset = args.dataset
num_chunks_list = args.num_chunks_list
epoch_list = args.epoch_list
num_samples = args.num_samples
epochs = args.epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if dataset == 'mnist':
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # Load the training and test datasets
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
elif dataset == 'cifar10':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loader
testloader = DataLoader(testset, batch_size=128)

acc_dict = {}
acc_file = f"chunk_analysis/{dataset}/acc_results_100.json"

for num_chunks in num_chunks_list:
    for epoch in epoch_list:
        # Initialize the network, optimizer, and loss function
        if dataset == 'mnist':
            net = Net().to(device)
            gen = F2U_GAN()
        elif dataset == 'cifar10':
            net = Net_Cifar().to(device)
            gen = F2U_GAN_CIFAR()
        optim = torch.optim.Adam(net.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        latent_dim = 128

        checkpoint_loaded = torch.load(f"./chunk_analysis/{dataset}/num_chunks{num_chunks}/checkpoint_epoch{epoch}.pth")
        gen.load_state_dict(checkpoint_loaded["gen_state_dict"])

        generated_dataset = GeneratedDataset(generator=gen.to("cpu"), num_samples=num_samples, latent_dim=latent_dim, num_classes=10, device="cpu")
        generated_dataloader = DataLoader(generated_dataset, batch_size=64, shuffle=True)

        for _ in range(epochs):
            for data in generated_dataloader:
                inputs, labels = data["image"].to(device), data["label"].to(device)
                optim.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optim.step()

        correct, loss = 0, 0.0
        net.eval()
        with torch.no_grad():
            for batch in testloader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        acc_dict[f"num_chunks_{num_chunks}_epoch_{epoch}"] = accuracy
        try:
            with open(acc_file, 'w', encoding='utf-8') as f:
                json.dump(acc_dict, f, ensure_ascii=False, indent=4) # indent makes it readable
            print(f"Accuracy from {num_chunks} num_chunks epoch {epoch} successfully saved to {acc_file}")
        except Exception as e:
            print(f"Error saving accuracy dict to JSON: {e}")
