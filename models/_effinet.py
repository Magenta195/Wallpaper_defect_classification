import torchvision.models as models
import torch.nn as nn


eff_model_dict = {
    'b0' : models.efficientnet_b0,
    'b1' : models.efficientnet_b1,
    'b2' : models.efficientnet_b2,
    'b3' : models.efficientnet_b3,
    'b4' : models.efficientnet_b4,
    'b5' : models.efficientnet_b5,
    'b6' : models.efficientnet_b6,
    'b7' : models.efficientnet_b7,
}
class BaseModel(nn.Module):
    def __init__(
        self, 
        num_classes : int = 19,
        pretrained : bool = True,
        model_num : str = 'b0',
        **kwargs
    ):
        super(BaseModel, self).__init__()
        self.backbone = eff_model_dict[ model_num ]( pretrained=pretrained )
        self.classifier = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
def _get_effinet_models(
        num_classes : int = 19,
        pretrained : bool = True,
        model_num : str = 'b0',
        **kwargs
    ) -> nn.Module :
    if model_num not in eff_model_dict :
        raise NotImplementedError("Invaild Model")
    
    return BaseModel(
        num_classes = num_classes,
        pretrained = pretrained,
        model_num = model_num,
        **kwargs,
    )


