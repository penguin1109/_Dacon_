import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, channel_in, patch_size, emb_size, img_size):
        super(PatchEmbed, self).__init__()
        """
        patch_size : 이미지를 patch단위로 자를때 한 변의 길이
        emb_size : 자른 이미지들을 embedding하려는 벡터의 크기
        img_size : (tuple) 입력되는 이미지의 크기
        """
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(channel_in, emb_size, kernel_size = patch_size, stride = patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.positions = nn.Parameter(torch.randn((img_size[0] // batch_size) * (img_size[1]// batch_size) + 1, emb_size))
    
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim = 1)
        x += self.positions
        
        return X
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 1280, num_heads: int = 16, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) 
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        
        return out
    
class ResAdd(nn.Module):
      def __init__(self, fn):
    super().__init__()
    self.fn = fn
  
  def forward(self, x, **kwargs):
    res = x
    x = self.fn(x, **kwargs)
    x += res

    return x

class FeedForwardBlock(nn.Sequential):
  def __init__(self, emb_size = 1280, expansion = 4, drop_p = 0):
    super().__init__(
        nn.Linear(emb_size, expansion * emb_size),
        nn.GELU(),
        nn.Dropout(drop_p),
        nn.Linear(expansion * emb_size, emb_size),
    )
    
class TransformerEncoderBlock(nn.Sequential):
      def __init__(self, emb_size : int = 1280, 
               drop_p = 0,
               forward_expansion = 4, forward_drop_p = 0, **kwargs):
    super().__init__(
        ResAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            MultiHeadAttention(emb_size, **kwargs),
            nn.Dropout(drop_p)
        )),
        ResAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            FeedForwardBlock(
                emb_size, expansion = forward_expansion, drop_p = forward_drop_p),
            nn.Dropout(drop_p)
        ))
    )

class TransformerEncoder(nn.Sequential):
  def __init__(self, depth, **kwargs):
    super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
    
# Head Layer for final Classification
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