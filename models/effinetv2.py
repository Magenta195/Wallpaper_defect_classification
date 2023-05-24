import torchvision.models as models
import torch.nn as nn


MODEL_DICT = {
    'v2_s' : models.efficientnet_v2_s,
    'v2_s_weight' : models.EfficientNet_V2_S_Weights,
    'v2_m' : models.efficientnet_v2_m,
    'v2_m_weight' : models.EfficientNet_V2_M_Weights,
    'v2_l' : models.efficientnet_v2_l,
    'v2_l_weight' : models.EfficientNet_V2_L_Weights,
}


class BaseModel(nn.Module):
    def __init__(
        self, 
        num_classes: int = 19,
        pretrained: bool = True,
        model_num: str = 'b0',
        **kwargs
    ):
        super(BaseModel, self).__init__()
        self.backbone = MODEL_DICT[model_num](
            weights = MODEL_DICT[model_num + '_weight'] if pretrained else None,
            **kwargs
        )
        self.classifier = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
    
def _get_effinetv2_models(
        num_classes: int = 19,
        pretrained: bool = True,
        model_num: str = 'v2_s',
        **kwargs
    ) -> nn.Module:
    """This function return CNN model"""
    if model_num not in MODEL_DICT:
        raise NotImplementedError("Invaild Model")
    
    return BaseModel(
        num_classes = num_classes,
        pretrained = pretrained,
        model_num = model_num,
        **kwargs,
    )