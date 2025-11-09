"""
将MoE集成到Transformer中的完整示例
演示如何在Encoder/Decoder中使用混合专家架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 从已有文件导入
from Transformer import MultiHeadAttention, PositionalEncoding
from MoE import MoEEfficient, LoadBalancingLoss


class EncoderLayerWithMoE(nn.Module):
    """使用MoE的编码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, num_experts=8, top_k=2, dropout=0.1):
        """
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: FFN隐藏层维度
            num_experts: MoE专家数量
            top_k: 每次激活的专家数
            dropout: Dropout比率
        """
        super(EncoderLayerWithMoE, self).__init__()
        
        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 混合专家层（替代标准FFN）
        self.moe = MoEEfficient(d_model, d_ff, num_experts, top_k, dropout)
        
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
        返回:
            x: 输出张量
            aux_loss: 辅助损失（用于负载均衡）
        """
        # 多头自注意力 + 残差连接 + Layer Norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # MoE层 + 残差连接 + Layer Norm
        moe_output = self.moe(x)
        x = self.norm2(x + self.dropout(moe_output))
        
        # 计算负载均衡损失
        gates, indices = self.moe.router(x)
        lb_loss_fn = LoadBalancingLoss(self.moe.num_experts)
        aux_loss = lb_loss_fn(gates, indices)
        
        return x, aux_loss


class DecoderLayerWithMoE(nn.Module):
    """使用MoE的解码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, num_experts=8, top_k=2, dropout=0.1):
        """
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: FFN隐藏层维度
            num_experts: MoE专家数量
            top_k: 每次激活的专家数
            dropout: Dropout比率
        """
        super(DecoderLayerWithMoE, self).__init__()
        
        # 掩码多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 跨注意力机制
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        # 混合专家层（替代标准FFN）
        self.moe = MoEEfficient(d_model, d_ff, num_experts, top_k, dropout)
        
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
            x: 解码器输入
            encoder_output: 编码器输出
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        返回:
            x: 输出张量
            aux_loss: 辅助损失
        """
        # 掩码自注意力 + 残差 + LN
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 跨注意力 + 残差 + LN
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # MoE层 + 残差 + LN
        moe_output = self.moe(x)
        x = self.norm3(x + self.dropout(moe_output))
        
        # 计算负载均衡损失
        gates, indices = self.moe.router(x)
        lb_loss_fn = LoadBalancingLoss(self.moe.num_experts)
        aux_loss = lb_loss_fn(gates, indices)
        
        return x, aux_loss


class TransformerWithMoE(nn.Module):
    """使用MoE的完整Transformer模型"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=2, num_decoder_layers=2, d_ff=2048,
                 num_experts=8, top_k=2, max_seq_length=5000, dropout=0.1,
                 aux_loss_weight=0.01):
        """
        参数:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            d_ff: FFN维度
            num_experts: MoE专家数量
            top_k: 每次激活的专家数
            max_seq_length: 最大序列长度
            dropout: Dropout比率
            aux_loss_weight: 辅助损失权重
        """
        super(TransformerWithMoE, self).__init__()
        
        # 词嵌入
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # 使用MoE的编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayerWithMoE(d_model, num_heads, d_ff, num_experts, top_k, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # 使用MoE的解码器层
        self.decoder_layers = nn.ModuleList([
            DecoderLayerWithMoE(d_model, num_heads, d_ff, num_experts, top_k, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.aux_loss_weight = aux_loss_weight
        
    def generate_mask(self, src, tgt):
        """生成掩码"""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        tgt_seq_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=tgt.device)).bool()
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        """
        前向传播
        参数:
            src: 源序列
            tgt: 目标序列
        返回:
            output: 输出logits
            aux_loss: 总的辅助损失
        """
        # 生成掩码
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 编码器
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        total_aux_loss = 0
        for encoder_layer in self.encoder_layers:
            x, aux_loss = encoder_layer(x, src_mask)
            total_aux_loss += aux_loss
        
        encoder_output = x
        
        # 解码器
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for decoder_layer in self.decoder_layers:
            x, aux_loss = decoder_layer(x, encoder_output, src_mask, tgt_mask)
            total_aux_loss += aux_loss
        
        # 输出层
        output = self.fc_out(x)
        
        # 返回输出和辅助损失
        return output, total_aux_loss * self.aux_loss_weight


# 使用示例
if __name__ == "__main__":
    print("=" * 80)
    print("MoE-Transformer集成测试")
    print("=" * 80)
    
    # 模型参数
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 2
    num_decoder_layers = 2
    d_ff = 2048
    num_experts = 8      # MoE专家数
    top_k = 2            # 每次激活2个专家
    
    # 创建标准Transformer（对比）
    print("\n1. 标准Transformer参数统计")
    print("-" * 80)
    
    # 简单计算：每层的FFN参数
    standard_ffn_params = d_model * d_ff * 2  # W1 + W2
    standard_attention_params = 4 * d_model * d_model  # Q, K, V, O
    standard_layer_params = standard_ffn_params + standard_attention_params
    standard_total = (num_encoder_layers + num_decoder_layers) * standard_layer_params
    
    print(f"每层FFN参数: {standard_ffn_params:,}")
    print(f"每层Attention参数: {standard_attention_params:,}")
    print(f"每层总参数: {standard_layer_params:,}")
    print(f"所有层总参数: {standard_total:,}")
    
    # MoE-Transformer参数
    print("\n2. MoE-Transformer参数统计")
    print("-" * 80)
    
    moe_ffn_params = num_experts * standard_ffn_params  # 8个专家
    moe_router_params = d_model * num_experts  # 路由网络
    moe_layer_params = moe_ffn_params + moe_router_params + standard_attention_params
    moe_total = (num_encoder_layers + num_decoder_layers) * moe_layer_params
    
    print(f"每层MoE参数: {moe_ffn_params:,} ({num_experts}个专家)")
    print(f"每层Router参数: {moe_router_params:,}")
    print(f"每层Attention参数: {standard_attention_params:,}")
    print(f"每层总参数: {moe_layer_params:,}")
    print(f"所有层总参数: {moe_total:,}")
    
    # 效率对比
    print("\n3. 效率对比")
    print("-" * 80)
    param_increase = moe_total / standard_total
    compute_increase = top_k / num_experts * param_increase
    
    print(f"参数量增加: {param_increase:.1f}x")
    print(f"计算量增加: {compute_increase:.1f}x")
    print(f"效率提升: {param_increase / compute_increase:.1f}x")
    print(f"(拥有{param_increase:.1f}x的参数，但只需要{compute_increase:.1f}x的计算)")
    
    # 创建实际模型
    print("\n4. 实际模型测试")
    print("-" * 80)
    
    model = TransformerWithMoE(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
        aux_loss_weight=0.01
    )
    
    # 创建示例输入
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # 前向传播
    with torch.no_grad():
        output, aux_loss = model(src, tgt)
    
    print(f"源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")
    print(f"输出形状: {output.shape}")
    print(f"辅助损失: {aux_loss.item():.6f}")
    print(f"模型总参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练示例
    print("\n5. 训练流程示例")
    print("-" * 80)
    print("""
    # 在训练循环中使用MoE-Transformer
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for batch in dataloader:
        src, tgt = batch
        
        # 前向传播
        output, aux_loss = model(src, tgt[:, :-1])  # teacher forcing
        
        # 计算主任务损失（交叉熵）
        main_loss = F.cross_entropy(
            output.reshape(-1, vocab_size),
            tgt[:, 1:].reshape(-1)
        )
        
        # 总损失 = 主损失 + 辅助损失（负载均衡）
        total_loss = main_loss + aux_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f'Main Loss: {main_loss.item():.4f}, Aux Loss: {aux_loss.item():.4f}')
    """)
    
    print("\n" + "=" * 80)
    print("💡 MoE在Transformer中的优势")
    print("=" * 80)
    print(f"✅ 参数容量提升 {num_experts}x，但推理时仅激活 {top_k}/{num_experts} 的专家")
    print(f"✅ 每个token可以被专门的专家处理，提升模型表达能力")
    print(f"✅ 计算量仅增加 {top_k}x，远小于参数增加的 {num_experts}x")
    print(f"✅ 特别适合超大规模模型（如GPT-4、Switch Transformer等）")
    print("=" * 80)
