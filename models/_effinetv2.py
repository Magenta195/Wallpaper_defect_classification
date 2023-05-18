import torchvision.models as models
import torch.nn as nn


eff_model_v2_dict = {
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
        num_classes : int = 19,
        pretrained : bool = True,
        model_num : str = 'b0',
        **kwargs
    ):
        super(BaseModel, self).__init__()
        self.backbone = eff_model_v2_dict[ model_num ]( weights = eff_model_v2_dict[ model_num + '_weight' ] if pretrained else None )
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    
def _get_effinetv2_models(
        num_classes : int = 19,
        pretrained : bool = True,
        model_num : str = 'v2_s',
        **kwargs
    ) -> nn.Module :
    if model_num not in eff_model_v2_dict :
        raise NotImplementedError("Invaild Model")
    
    return BaseModel(
        num_classes = num_classes,
        pretrained = pretrained,
        model_num = model_num,
        **kwargs,
    )


