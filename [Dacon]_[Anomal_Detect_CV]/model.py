class BaseModel(nn.Module):
    def __init__(self, num_classes=1000, drop_p=0., pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        # self.model = timm.create_model('tf_efficientnet_b5_ns', pretrained=pretrained, num_classes=0, drop_rate=drop_p,global_pool = '')
        # self.model = timm.create_model('tf_efficientnetv2_xl_in21k', pretrained = pretrained, num_classed = num_classes, drop_rate = drop_p)
        # self.model = timm.create_model('tf_efficientnet_b5_ap', pretrained = pretrained, num_classes =num_classes, drop_rate = drop_p)
        self.model = timm.create_model('tf_efficientnet_b7_ns', pretrained = pretrained, num_classes = num_classes)
        # self.model = timm.create_model('vit_large_patch32_384', pretrained = pretrained, num_classes = num_classes)
        # self.model = timm.create_model('resnetv2_101x1_bitm', pretrained = pretrained, num_classes = num_classes, drop_rate = drop_p)
        #self.model = timm.create_model('resnetv2_50x3_bitm', pretrained = pretrained, num_classes = num_classes, drop_rate = drop_p)
            
    @torch.cuda.amp.autocast()
    def forward(self, img):
        return self.model(img)

class SELayer(nn.Module):
  def __init__(self, ch_in, ratio):
    super().__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
        nn.Linear(ch_in, ch_in//ratio, bias = False),
        nn.ReLU(inplace = True),
        nn.Linear(ch_in // ratio, ch_in, bias = False),
        nn.Sigmoid()
    )
  def forward(self, x):
    b, c, _, _ = x.size() ## batchsize, channel
    y = self.avg_pool(x).view(b,c)
    y = self.fc(y).view(b,c,1,1)
    return x * y.expand_as(x)


class ClassHeadClassifier(nn.Module):
    def __init__(self,with_se=False,  num_base_features=1000, num_classes=15, drop_p=0.1):
        super().__init__()
        self.with_se = with_se
        self.pool = False
        if (with_se == True):
          self.se = SELayer(1280, ratio = 4)
          self.conv = nn.Conv2d(in_channels = 1280, out_channels = 1280, kernel_size = 1)
          self.pool = nn.AdaptiveAvgPool2d(1)
          self.norm = nn.BatchNorm2d(num_features = 1280)
          self.classifier = nn.Linear(1280, num_classes)
        elif self.pool:
          self.pool = nn.AdaptiveAvgPool2d(1)
          self.classifier = nn.Linear(2048, num_classes)
        else:
          self.classifier = nn.Linear(num_base_features, num_classes)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_p)
    
    @torch.cuda.amp.autocast()
    def forward(self, x):
      if self.with_se == True:
        se = self.se(x)
        x = (self.conv(x) + se)
        x = self.pool(self.norm(x))
      elif self.pool:
        x = self.pool(x)
      else:
        return self.classifier(self.act(self.drop(x)))
      b,c,_,_ = x.size()
      x = x.view(b,c)
      return self.classifier(self.act(self.drop(x)))


class ClassClassifier(nn.Module):
    def __init__(self, num_base_features=1000,num_classes=15, drop_p=0.1):
        super().__init__()
        self.classifier = timm.create_model('tf_efficientnet_b0_ns', pretrained = True, num_classes = num_classes)
        
    def forward(self, x, not_use=None):
        return self.classifier(x)

class AnomalHeadClassifier(nn.Module):
    def __init__(self, with_se = False, num_base_features=1000, num_classes=49, drop_p=0.1):
        super().__init__()
        self.with_se = with_se
        self.pool = False
        if (with_se == True):
          self.se = SELayer(2048, ratio = 16)
          self.conv = nn.Conv2d(in_channels = 2048, out_channels = 2048, kernel_size = 1)
          self.pool = nn.AdaptiveAvgPool2d(1)
          self.norm = nn.BatchNorm2d(num_features = 2048)
          self.classifier = nn.Linear(2048, num_classes)
        elif self.pool:
          self.pool = nn.AdaptiveAvgPool2d(1)
          self.classifier = nn.Linear(2048, num_classes)
        else:
          self.classifier = nn.Linear(num_base_features, num_classes)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_p)
    
    @torch.cuda.amp.autocast()
    def forward(self, x,mask):
      if self.with_se == True:
        se = self.se(x)
        x = (self.conv(x) + se)
        x = self.pool(self.norm(x))
      elif self.pool:
        x = self.pool(x)
      else:
        x = self.classifier(self.act(self.drop(x)))
        return x.masked_fill_(mask, -10000,)
      b,c,_,_ = x.size()
      x = x.view(b,c)
      x = self.classifier(self.act(self.drop(x)))
      x.masked_fill_(mask, -10000.)
      return x


class AnomalClassifier(nn.Module):
    def __init__(self, num_base_features=1000, num_classes=49, drop_p=0.1):
        super().__init__()
        self.classifier = timm.create_model('tf_efficientnet_b0_ns', pretrained = True, num_classes = num_classes)
        
    def forward(self, x, mask):
        x = self.classifier(x)
        x.masked_fill_(mask, -10000,)
        return x

