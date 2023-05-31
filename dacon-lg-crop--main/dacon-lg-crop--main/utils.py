# 2D image -> sequence of flattened image patches
class PatchEmbed(nn.Module):
  def __init__(self, channel_in, patch_size, emb_size, img_size):
    super().__init__()
    """
    channel_in : 3
    patch_size : 전체 이미지를 조각 낼때 하나의 조각의 한변 길이
    emb_size : 각각의 patch image를 embedding을 했을 떄의 길이
    img_size : 입력할 reshape된 image size (patch의 개수를 위해 필요)
    """
    self.patch_size = patch_size
    """
    # Sol1.
    self.projection = nn.Sequential(
        # break the image in stride1 * stride2 patches and flatten them
        Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1 = self.patch_size, s2 = self.patch_size),
        nn.Linear(patch_size * patch_size * channel_in, emb_size)
    )
    """
    # Sol2 -> Improvement in performance
    self.projection =  nn.Sequential(
        # using a conv layer instead of a linear layer for performance gain
        nn.Conv2d(channel_in, emb_size, kernel_size = patch_size, stride = patch_size),
        Rearrange('b e (h) (w) -> b (h w) e')
    )

    # cls token은 randomly initialized된 torch Parameter이다.
    # class token (= learnable embedding "classification token")
    self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))

    # position embedding (None, 1d, 2d, relative)
    self.positions = nn.Parameter(torch.randn((img_size[0] // patch_size) * (img_size[1] // patch_size) + 1, emb_size))
  
  def forward(self, x):
    b, _, _, _ = x.shape
    # batch의 개수만큼 cls token을 추가하는 과정을 반복
    x = self.projection(x)
    cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
    x = torch.cat([cls_tokens, x], dim = 1)
    # add position embedding
    x += self.positions

    return x
  
  
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 1280, num_heads: int = 16, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
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
