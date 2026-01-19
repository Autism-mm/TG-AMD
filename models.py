from torch import nn
from torchvision import models

class ResNet_34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_34, self).__init__()
        self.vgg = models.vgg16_bn(pretrained=True)
        # 保留原有的分类层
        self.vgg.classifier[6] = nn.Linear(4096, 1000)
        # 新加一层
        self.extra_fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        # print(f"[Forward] Running on {x.device}")
        x = self.vgg(x)  # [batch_size, 1000]
        x = self.extra_fc(x)  # [batch_size, 1000 * num_classes]
        return x

class SARATR_X(nn.Module):
    def __init__(self, num_classes, input_dim=256, nhead=8, num_layers=6):
        super(SARATR_X, self).__init__()
        # 新增：把3通道投影到input_dim
        self.patch_embed = nn.Linear(3, input_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: [batch, 3, 224, 224]
        batch, c, h, w = x.shape
        # 1. 展平空间维度
        x = x.permute(0, 2, 3, 1)  # [batch, h, w, c]
        x = x.reshape(batch, h * w, c)  # [batch, seq_len, c]
        # 2. 投影到input_dim
        x = self.patch_embed(x)  # [batch, seq_len, input_dim]
        # 3. 变换维度以适配transformer
        x = x.permute(1, 0, 2)  # [seq_len, batch, input_dim]
        # 4. transformer
        x = self.transformer(x)
        x = x.mean(dim=0)  # 池化
        x = self.fc(x)
        return x
    
def HiViT_base(classes):
    model = HiViT(
        img_size=224, embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., in_chans=3, mlp_ratio=4.,
        num_classes=classes,
        ape=True, rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    # 不加载预训练权重，不冻结参数，直接返回模型
    return model