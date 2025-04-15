import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ViT import SimpleViT
from Data import test_dataset


model = SimpleViT(patch_dim=48, num_patches=64, dim=128)
model.load_state_dict(torch.load("/home/rajdeep/UMC203project/Vision_transformer/90-47.pth"))
model.eval()


test_loader = test_dataset(patch_size=4)


# Initialize trackers
all_preds = []
all_labels = []

# Inference loop
model.eval()
with torch.no_grad():
    for batch in test_loader:
        x, labels = batch

        outputs = model(x)                     # [B, num_classes]
        preds = torch.argmax(outputs, dim=1)   # [B]

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
        print("done")

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate metrics
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='weighted')
rec = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
cm = confusion_matrix(all_labels, all_preds)

# Print results
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Print confusion matrix as a numpy array
print("\nConfusion Matrix:\n", cm)

# Save the confusion matrix plot
plt.figure(figsize=(10, 8))
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_train2.png")  # Save instead of show
plt.close()  # Close the plot to prevent it from displaying
