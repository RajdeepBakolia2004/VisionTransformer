from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from SimpleViT import SimpleViT
import torch
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import seaborn as sns

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
])


train_dataset = ImageFolder(root='/home/rajdeep/UMC203project/SimpleTransformer/extract/train', transform=transform)
test_dataset = ImageFolder(root='/home/rajdeep/UMC203project/SimpleTransformer/extract/test', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def patching(X):
    patch_size = 4
    stride = 4
    X = X.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    X = X.permute(0, 2, 3, 1, 4, 5)
    X = X.reshape(X.shape[0] ,-1, patch_size * patch_size )
    return X


def epoch_vs_accuracy(device):
    import matplotlib.pyplot as plt
    import torch

    accuracy_test = []
    accuracy_train = []

    for i in range(1, 31):
        # Initialize model and load weights
        vit = SimpleViT(patch_dim=16, num_patches=49, dim=64).to(device)
        checkpoint_path = f'/home/rajdeep/UMC203project/SimpleTransformer/simplevit_epoch{i}.pth'
        vit.load_state_dict(torch.load(checkpoint_path))
        vit.eval()

        correct_test = 0
        total_test = 0
        correct_train = 0
        total_train = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                patches = patching(images)  # ensure patching returns correct shape
                outputs = vit(patches)

                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

        acc = 100 * correct_test / total_test
        accuracy_test.append(acc)
        print(f"Test Accuracy for epoch {i}: {acc:.2f}%")
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                patches = patching(images)
                outputs = vit(patches)
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()
        acc_train = 100 * correct_train / total_train
        accuracy_train.append(acc_train)
        print(f"Train Accuracy for epoch {i}: {acc_train:.2f}%")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 31), accuracy_test, label='Test Accuracy')
    plt.plot(range(1, 31), accuracy_train, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Epoch vs Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('epoch_vs_accuracy.png')
    plt.close()

def metrics(model, loader, device,filename):
    all_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            patches = patching(images)
            outputs = model(patches)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    with open(filename, 'w') as f:
        f.write("Accuracy: {:.2f}%\n".format(accuracy * 100))
        f.write("Precision: {:.2f}\n".format(precision))
        f.write("Recall: {:.2f}\n".format(recall))
        f.write("F1 Score: {:.2f}\n".format(f1))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\nClassification Report:\n")
        for label, metrics in report.items():
            f.write(f"Label {label}: {metrics}\n")
    # Save the confusion matrix plot
    plt.figure(figsize=(10, 8))
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{filename}.png")  # Save instead of show
    plt.close()  # Close the plot to prevent it from displaying

model = SimpleViT(patch_dim=16, num_patches=49, dim=64).to(device)
checkpoint_path = f'/home/rajdeep/UMC203project/SimpleTransformer/simplevit_epoch30.pth'
model.load_state_dict(torch.load(checkpoint_path))
metrics(model, test_loader, device,'2se0.txt')




