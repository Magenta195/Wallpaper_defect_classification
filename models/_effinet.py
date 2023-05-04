import torchvision.models as models
import torch.nn as nn


eff_model_dict = {
    'b0' : models.efficientnet_b0,
    'b0_weight' : models.EfficientNet_B0_Weights,
    'b1' : models.efficientnet_b1,
    'b1_weight' : models.EfficientNet_B1_Weights,
    'b2' : models.efficientnet_b2,
    'b2_weight' : models.EfficientNet_B2_Weights,
    'b3' : models.efficientnet_b3,
    'b3_weight' : models.EfficientNet_B3_Weights,
    'b4' : models.efficientnet_b4,
    'b4_weight' : models.EfficientNet_B4_Weights,
    'b5' : models.efficientnet_b5,
    'b5_weight' : models.EfficientNet_B5_Weights,
    'b6' : models.efficientnet_b6,
    'b6_weight' : models.EfficientNet_B6_Weights,
    'b7' : models.efficientnet_b7,
    'b7_weight' : models.EfficientNet_B7_Weights,
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
        self.backbone = eff_model_dict[ model_num ]( weights = eff_model_dict[ model_num + '_weight' ] if pretrained else None )
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


