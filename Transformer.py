"""
Transformer最小可用实现
包含多头注意力、跨注意力、残差网络等核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads):
        """
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Q, K, V的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出的线性变换层
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        参数:
            Q: Query矩阵 [batch_size, num_heads, seq_len, d_k]
            K: Key矩阵 [batch_size, num_heads, seq_len, d_k]
            V: Value矩阵 [batch_size, num_heads, seq_len, d_k]
            mask: 掩码矩阵
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        
        # 计算输出
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        参数:
            query: Query输入 [batch_size, seq_len, d_model]
            key: Key输入 [batch_size, seq_len, d_model]
            value: Value输入 [batch_size, seq_len, d_model]
            mask: 掩码矩阵
        """
        batch_size = query.size(0)
        
        # 线性变换并分割成多头
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        # [batch_size, num_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最后的线性变换
        output = self.W_o(attn_output)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        """
        参数:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 应用sin和cos函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
        """
        # 添加位置编码
        x = x + self.pe[:, :x.size(1), :]
        return x


class FeedForward(nn.Module):
    """前馈神经网络（带残差连接）"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        参数:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout比率
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
        """
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """编码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout比率
        """
        super(EncoderLayer, self).__init__()
        
        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 前馈神经网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 掩码矩阵
        """
        # 多头自注意力 + 残差连接 + Layer Norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + Layer Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """解码器层（包含跨注意力机制）"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout比率
        """
        super(DecoderLayer, self).__init__()
        
        # 掩码多头自注意力（Masked Self-Attention）
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 跨注意力机制（Cross-Attention）
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        # 前馈神经网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播
        参数:
            x: 解码器输入 [batch_size, tgt_seq_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_seq_len, d_model]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（用于掩码自注意力）
        """
        # 掩码多头自注意力 + 残差连接 + Layer Norm
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 跨注意力机制 + 残差连接 + Layer Norm
        # Query来自解码器，Key和Value来自编码器
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 前馈网络 + 残差连接 + Layer Norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    """完整的Transformer模型（2层Encoder + 2层Decoder）"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=2, num_decoder_layers=2, d_ff=2048, 
                 max_seq_length=5000, dropout=0.1):
        """
        参数:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            d_ff: 前馈网络隐藏层维度
            max_seq_length: 最大序列长度
            dropout: Dropout比率
        """
        super(Transformer, self).__init__()
        
        # 词嵌入层
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # 编码器层（2层）
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # 解码器层（2层）
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
    def generate_mask(self, src, tgt):
        """
        生成掩码矩阵
        参数:
            src: 源序列 [batch_size, src_seq_len]
            tgt: 目标序列 [batch_size, tgt_seq_len]
        """
        # 源序列填充掩码（可选，如果有padding的话）
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_seq_len]
        
        # 目标序列的掩码（用于掩码自注意力，防止看到未来的token）
        tgt_seq_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=tgt.device)).bool()
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, tgt_seq_len, tgt_seq_len]
        
        return src_mask, tgt_mask
        
    def encode(self, src, src_mask):
        """
        编码器前向传播
        参数:
            src: 源序列 [batch_size, src_seq_len]
            src_mask: 源序列掩码
        """
        # 词嵌入 + 位置编码
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 通过所有编码器层
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)
            
        return x
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        """
        解码器前向传播
        参数:
            tgt: 目标序列 [batch_size, tgt_seq_len]
            encoder_output: 编码器输出
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        """
        # 词嵌入 + 位置编码
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 通过所有解码器层
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, src_mask, tgt_mask)
            
        return x
    
    def forward(self, src, tgt):
        """
        前向传播
        参数:
            src: 源序列 [batch_size, src_seq_len]
            tgt: 目标序列 [batch_size, tgt_seq_len]
        返回:
            output: 输出logits [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # 生成掩码
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 编码
        encoder_output = self.encode(src, src_mask)
        
        # 解码
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # 输出层
        output = self.fc_out(decoder_output)
        
        return output


class EncoderLayerWithMoE(nn.Module):
    """使用MoE替代FFN的编码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, num_experts=8, top_k=2, dropout=0.1):
        """
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            num_experts: MoE专家数量
            top_k: 每次激活的专家数
            dropout: Dropout比率
        """
        super(EncoderLayerWithMoE, self).__init__()
        
        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 混合专家层（替代标准FFN）
        # 需要从MoE.py导入：from MoE import MoEEfficient
        # 这里使用简化版本，实际使用时应导入完整实现
        self.moe = self._create_simple_moe(d_model, d_ff, num_experts, top_k, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def _create_simple_moe(self, d_model, d_ff, num_experts, top_k, dropout):
        """创建简化的MoE（实际应用中请导入完整版本）"""
        # 这里返回标准FFN作为占位符
        # 实际使用: from MoE import MoEEfficient
        # return MoEEfficient(d_model, d_ff, num_experts, top_k, dropout)
        return FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 掩码矩阵
        """
        # 多头自注意力 + 残差连接 + Layer Norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # MoE层 + 残差连接 + Layer Norm
        moe_output = self.moe(x)
        x = self.norm2(x + self.dropout(moe_output))
        
        return x


# 使用示例
if __name__ == "__main__":
    # 设置参数
    src_vocab_size = 5000  # 源语言词汇表大小
    tgt_vocab_size = 5000  # 目标语言词汇表大小
    d_model = 512          # 模型维度
    num_heads = 8          # 注意力头数
    num_encoder_layers = 2 # 编码器层数
    num_decoder_layers = 2 # 解码器层数
    d_ff = 2048           # 前馈网络维度
    max_seq_length = 100   # 最大序列长度
    dropout = 0.1          # Dropout比率
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout
    )
    
    # 创建示例输入
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    # 随机生成源序列和目标序列（token索引）
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # 前向传播
    output = model(src, tgt)
    
    print("=" * 60)
    print("Transformer模型测试")
    print("=" * 60)
    print(f"源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)
    print("\n模型结构:")
    print(model)
