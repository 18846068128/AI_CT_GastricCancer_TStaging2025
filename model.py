import torch.nn as nn
import torchvision.models as models

def initialize_model(model_name, num_classes):
    model_map = {
        "resnet152": (models.resnet152, "fc"),
        "densenet169": (models.densenet169, "classifier"),
        "vgg19": (models.vgg19, "classifier.6")
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unsupported model: {model_name}")

    model_class, layer_path = model_map[model_name]
    model = model_class(pretrained=True)
    
    # 动态获取全连接层
    layers = layer_path.split('.')
    layer = model
    for l in layers:
        if l.isdigit():
            layer = layer[int(l)]
        else:
            layer = getattr(layer, l)
    
    in_features = layer.in_features
    if "classifier" in layer_path:
        layer = nn.Linear(in_features, num_classes)
    else:
        layer = nn.Linear(in_features, num_classes)
    
    return model.to("cuda" if torch.cuda.is_available() else "cpu")
