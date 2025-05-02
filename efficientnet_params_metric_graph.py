import torch
from torchvision import models
import matplotlib.pyplot as plt

params_list = []
metric_list = [0.347, 0.3072, 0.2986, 0.2965, 0.2755, 0.1711, 0.1113, 0.08238] # Val Accuracy
model_names = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
             'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
             'efficientnet_b6', 'efficientnet_b7']
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
             'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
             'efficientnet_b6', 'efficientnet_b7']:
    model = models.get_model(name, weights=None, num_classes=200)
    #num_params = count_parameters(model) / 1e6  # million 단위
    num_params = count_parameters(model) 
    params_list.append(num_params)
    
plt.figure(figsize=(10, 6))
plt.plot(params_list, metric_list, marker='o', linestyle='-', color='crimson')

for i, name in enumerate(model_names):
    # plt.text(params_million[i] + 0.8, accuracies[i] - 0.4, name)
    pass
plt.xlabel('Number of Parameters (Millions)')
plt.ylabel('Validation Accuracy (%)')
plt.title('Size vs Accuracy Trade-off: EfficientNet-B0~B7')
plt.grid(True)
plt.tight_layout()
plt.show()
