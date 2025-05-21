import torch
import torch.nn as nn
import torchvision.models as models

def initialize_model(model_name, num_classes):
    model_map = {
        "resnet152": (models.resnet152, "fc"),
        "densenet169": (models.densenet169, "classifier"),
        "vgg19": (models.vgg19, "classifier")
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unsupported model: {model_name}")

    model_class, layer_name = model_map[model_name]
    model = model_class(pretrained=True)
    
    # 处理不同模型结构
    if model_name == "resnet152":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "densenet169":
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == "vgg19":
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    return model.to("cuda" if torch.cuda.is_available() else "cpu")

