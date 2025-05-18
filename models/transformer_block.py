import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
This is a small manual implementation of transformer block,
although this is already a built-in block of torch.nn, but 
manually implementation helps understanding the design of
this dominated architecture.
'''

class MultiheadAttention(nn.Module):
    '''
    Here, we assume that we have an input of a matrix
    [B, L, D] where B is the number of batch size, L is the
    length of sequence, D is the embedding dim
    '''
    def __init__(self, d_input, n_heads, dropout):
        super().__init__()
        assert d_input % n_heads == 0   
        self.d_in_head = d_input // n_heads
        self.n_heads = n_heads
        self.d_input = d_input
        self.Q = nn.Linear(d_input, d_input)
        self.K = nn.Linear(d_input, d_input)
        self.V = nn.Linear(d_input, d_input)
        self.output = nn.Linear(d_input, d_input)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x): 
        '''
        x is the input of [B, L, D], we gonna multiply it to get
        out Q, K, V and then split them into n heads
        '''
        B = x.size(0)
        Q = self.Q(x) 
        K = self.K(x)
        V = self.V(x)
        '''
        We then split them into n_heads. Therefore we get matrices
        of size B, L, n_head, d_input // n_head  and then we transpose
        it to get B, n_head, L, d_input // n_head
        '''
        Q = Q.view(B, -1, self.n_heads, self.d_in_head).transpose(1, 2)
        K = K.view(B, -1, self.n_heads, self.d_in_head).transpose(1, 2).transpose(2, 3)
        V = V.view(B, -1, self.n_heads, self.d_in_head).transpose(1, 2)
        '''
        Then we compute Q multiply K to get our similarity matrix
        '''
        sim_matrix = torch.matmul(Q, K) # B, N, L, L
        sim_score = F.softmax(sim_matrix / math.sqrt(self.d_in_head), dim=3) # B, N, L, L
        sim_score = self.dropout(sim_score)
        output = torch.matmul(sim_score, V).transpose(1, 2) # B, L, N, D
        output = self.output(output.reshape(B, -1, self.d_input))
        return output # B, L, d_input
    
    
class FeedForward(nn.Module):
    '''
    This should be a custom class, for simplicity, I just wrote a simple
    neural net here
    '''
    def __init__(self, d_input, dropout=0.1):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_input, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_input)
        )
    def forward(self, x):
        return self.linear(x)
        
        
        
class Transformer_block(nn.Module):
    def __init__(self, d_input, n_heads=8, dropout=0.1, feedforward=None):
        super().__init__()
        self.Multihead = MultiheadAttention(d_input, n_heads, dropout)
        self.layernorm1 = nn.LayerNorm(d_input)
        if feedforward is None:
            self.FeedForward = FeedForward(d_input)
        else:
            self.FeedForward = feedforward
        self.layernorm2 = nn.LayerNorm(d_input)
        
    '''
    To deal with the encoder block, we input data through a multihead attention,
    layer norm the output
    and add a residual connection, then we put it through a FeedForward module
    that can be a custom class, 
    layer norm the output, 
    and add a residual connection at last
    '''
    def forward(self, x):
        x1 = self.Multihead(x)
        x1 = self.layernorm1(x1)
        x2 = x1 + x
        x3 = self.FeedForward(x2)
        x3 = self.layernorm2(x3)
        x4 = x3 + x2
        return x4
    
class ViT(nn.Module):
    '''
    Here we assume that the input pictures are all with same H and W,
    the module here wants an input of B, C, H, W, the raw format of a
    picture, and gives an output B, patches, embed_dim. For downstream
    task like classification, a MLP head is needed.
    
    Note: H and W have to be same
    '''
    def __init__(self, height, patch_size, embed_dim=256, num_transformer=5):
        super().__init__()
        assert height % patch_size == 0
        self.patch_size = patch_size
        self.patch_to_T = (height // patch_size) ** 2
        self.patch_dim = patch_size ** 2 * 3
        self.embed_patch_layer = nn.Linear(self.patch_dim, embed_dim)
        
        self.CLS_token = nn.Parameter(torch.randn([1, 1, embed_dim]))
        self.pos_encode = nn.Parameter(torch.randn([1, self.patch_to_T + 1, embed_dim]))
        '''
        The positional encoding part is required, and a unique CLS token is added, which 
        reflects some overall features
        '''
        self.transformers = nn.ModuleList([
            Transformer_block(embed_dim) for _ in range(num_transformer)
        ])
        
    def forward(self, x):
        '''
        x of the shape B, C, H, W, I need to unfold it
        '''
        B = x.size(0)
        x = x.unfold(2, self.patch_size, self.patch_size).permute(0, 4, 1, 2, 3).contiguous()
        x = x.unfold(4, self.patch_size, self.patch_size).permute(0, 1, 5, 2, 3, 4).contiguous()
        x = torch.flatten(x, 3).reshape(B, -1, self.patch_dim)
        embed_x = self.embed_patch_layer(x)
        embed_x = torch.cat([embed_x, self.CLS_token.expand(B, -1, -1)], dim=1)
        embed_x = embed_x + self.pos_encode # Now B, num_patch + 1, embed_dim
        for transformer in self.transformers:
            embed_x = transformer(embed_x)
        return embed_x
    
def main():
    x = torch.randn([3, 3, 64, 64])
    model = ViT(64, 16)
    y = model(x)
    print(y.size())
    
if __name__ == '__main__':
    main()
        
        