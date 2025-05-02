import torch
from torchvision import models
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis
import numpy as np

model_names = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
               'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
               'efficientnet_b6', 'efficientnet_b7']
val_accuracies = [0.347, 0.3072, 0.2986, 0.2965, 0.2755, 0.1711, 0.1113, 0.08238]  # Validation Accuracy

param_list = []
flop_list = []

# 입력 샘플 (TinyImageNet 크기: 64x64)
dummy_input = torch.randn(1, 3, 64, 64)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for name in model_names:
    model = models.get_model(name, weights=None, num_classes=200)
    model.eval()
    param_list.append(count_parameters(model) / 1e6)  
    flop_analysis = FlopCountAnalysis(model, dummy_input)
    flop_list.append(flop_analysis.total() / 1e9)  

# 각 점마다 색과 마커 다르게
markers = ['o', 's', '^', 'v', 'D', '*', 'X', 'P']
colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))

plt.figure(figsize=(10, 6))
for i in range(len(model_names)):
    plt.scatter(param_list[i], val_accuracies[i], marker=markers[i], color=colors[i], label=model_names[i])
plt.xlabel('Number of Parameters (Millions)')
plt.ylabel('Validation Accuracy')
plt.title('Model Size vs Validation Accuracy (EfficientNet-B0~B7)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i in range(len(model_names)):
    plt.scatter(flop_list[i], val_accuracies[i], marker=markers[i], color=colors[i], label=model_names[i])
plt.xlabel('FLOPs (GigaFLOPs)')
plt.ylabel('Validation Accuracy')
plt.title('FLOPs vs Validation Accuracy (EfficientNet-B0~B7)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
