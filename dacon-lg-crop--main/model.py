class ClassificationHead(nn.Module):
  def __init__(self, emb_size, n_classes):
    super().__init__()
    self.reduce = Reduce('b n e -> b e', reduction = 'mean')
    self.norm = nn.LayerNorm(emb_size)
    self.fc = nn.Linear(emb_size, n_classes)
  
  def forward(self, vit_enc, seq, cnn_enc):
    # seq는 환경 데이터를 바탕으로 하는 시계열 바탕의 데이터이다.
    vit_enc = self.reduce(vit_enc)
    out = torch.cat([vit_enc, seq, cnn_enc], dim = 1)
    out = self.norm(out)
    out = self.fc(out)
    return out

class CNNEncoder(nn.Module):
  def __init__(self):
    super(CNNEncoder, self).__init__()
    # 먼저 새로운 이미지로 pretrain을 해서 pretrained weights를 넣어줌
    self.model = models.efficientnet_b5(pretrained = True).features

  def forward(self, inputs):
    output = self.model(inputs)
    return output

class RNNDecoder(nn.Module):
  def __init__(self, max_len, embedding_dim, num_features):
    super(RNNDecoder, self).__init__()
    self.conv = nn.Conv1d(num_features,num_features*2, kernel_size = 1,stride = 1,dilation = 1,padding = 0)
    self.bn = nn.BatchNorm1d(num_features*2)

    self.lstm = nn.LSTM(max_len, embedding_dim)
    self.rnn_fc = nn.Linear(num_features *2* embedding_dim, embedding_dim)
    

  def forward(self, dec_in):
    # RNN모델에 넣어줄 데이터는
    dec_in = self.bn(self.conv(dec_in))
    hidden, _ = self.lstm(dec_in)
    hidden = hidden.view(hidden.size(0), -1)
    hidden = self.rnn_fc(hidden)
    return hidden

class ViT_Mix(nn.Module):
  def __init__(self, channel_in = 3,
               patch_size = 16,
               emb_size = 1280,
               rnn_emb_size = 980,
               depth = 16,
               max_len = cfg['max_len'],
               **kwargs):
    super().__init__()

    self.cnn_encoder = CNNEncoder()
    self.cnn_pool = nn.AdaptiveAvgPool2d(1)
    self.cnn_fc = nn.Linear(2048, 1024)

    self.emb = PatchEmbed(channel_in, patch_size, emb_size, cfg['vit_img'])
    self.enc = TransformerEncoder(depth, emb_size = emb_size, **kwargs)
    self.rnn = RNNDecoder(max_len, rnn_emb_size, num_features = cfg['num_feats'])

    self.final_classifier = ClassificationHead(emb_size + rnn_emb_size + 1024, cfg['num_class'])

  def forward(self, eff_img, vit_img, seq):
    # predicted by EfficientNet_b5
    cnn = self.cnn_pool(self.cnn_encoder(eff_img))
    cnn = cnn.view(cnn.size(0), -1)
    cnn = self.cnn_fc(cnn)

    # predicted by ViT
    enc = self.enc(self.emb(vit_img)) 
    
    # predicted by RNN
    seq = self.rnn(seq)

    final = self.final_classifier(enc, seq, cnn)


    return final
  
def initialize_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.kaiming_uniform_(m.weight.data)
    nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.LayerNorm):
    nn.init.constant_(m.weight.data, 1)
    nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Conv1d):
    nn.init.kaiming_uniform_(m.weight.data)
    if m.bias is not None:
      nn.init.constant_(m.bias.data, 0)
