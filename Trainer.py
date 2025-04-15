
import torch
import torch.nn as nn
import torch.optim as optim
from ViT import SimpleViT
from Data import train_dataset

train_loader = train_dataset(patch_size=4)

# Define device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 10
learning_rate = 3e-4  # Standard LR for transformers
weight_decay = 1e-2   # Regularization
batch_size = 64

# Move model to device
vit = SimpleViT(patch_dim=48, num_patches=64, dim=128).to(device)

# Loss function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(vit.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning Rate Scheduler (cosine decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training Loop
for epoch in range(num_epochs):
    vit.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = vit(images)
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Update learning rate
    scheduler.step()

    # Print stats
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

# Save the trained model
torch.save(vit.state_dict(), "vit_model2.pth")
print("Model saved successfully!")
