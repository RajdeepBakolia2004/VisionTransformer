from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from SimpleViT import SimpleViT
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm



train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),          # rotate slightly
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # small shifts
    transforms.RandomHorizontalFlip(),              # rarely used, but can help
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='', train=True, transform=train_transform, download=False)
test_dataset = datasets.MNIST(root='', train=False, transform=test_transform, download=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def patching(X):
    patch_size = 4
    stride = 4
    X = X.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    X = X.permute(0, 2, 3, 1, 4, 5)
    X = X.reshape(X.shape[0] ,-1, patch_size * patch_size )
    return X


# Hyperparameters
num_epochs = 30
warmup_epochs = 5
learning_rate = 3e-4
weight_decay = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vit = SimpleViT(patch_dim=16, num_patches=49, dim=64).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(vit.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Scheduler (after warmup)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)

def evaluate(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            patches = patching(images)
            outputs = model(patches)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100 * correct / total

warmup_epochs = 5
base_lr = learning_rate

def linear_warmup(current_epoch, total_warmup_epochs, base_lr):
    return float(current_epoch + 1) / float(total_warmup_epochs)

for epoch in range(num_epochs):
    vit.train()
    running_loss = 0.0
    correct = 0
    total = 0
    if epoch < warmup_epochs:
        warmup_factor = linear_warmup(epoch, warmup_epochs, base_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * warmup_factor
    else:
        scheduler.step()  
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        patches = patching(images)

        outputs = vit(patches)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        running_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item(), acc=100 * correct / total)
    train_acc = evaluate(vit, train_loader, device)
    test_acc = evaluate(vit, test_loader, device)
    print(f"\n✅ Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%, LR = {optimizer.param_groups[0]['lr']:.6f}")
    torch.save(vit.state_dict(), f"simplevit_epoch{epoch+1}.pth")