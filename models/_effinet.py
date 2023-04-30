import torchvision.models as models
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(
        self, 
        num_classes : int = 10,
        pretrained : bool = True,
        **kwargs
    ):
        super(BaseModel, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        self.classifier = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x