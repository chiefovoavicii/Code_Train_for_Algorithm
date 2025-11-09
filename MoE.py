"""
混合专家模型(Mixture of Experts, MoE)最小实现
用于替代Transformer中的FFN层，大幅减少计算量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """单个专家网络（等价于标准FFN）"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        参数:
            d_model: 输入/输出维度
            d_ff: 隐藏层维度
            dropout: Dropout比率
        """
        super(Expert, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        参数:
            x: [batch_size, seq_len, d_model] 或 [num_tokens, d_model]
        """
        # FFN: W2(Dropout(ReLU(W1(x))))
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Router(nn.Module):
    """路由网络（门控网络）- 决定每个token由哪些专家处理"""
    
    def __init__(self, d_model, num_experts, top_k=2):
        """
        参数:
            d_model: 输入维度
            num_experts: 专家总数
            top_k: 每个token选择的专家数量
        """
        super(Router, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 门控网络：将输入映射到专家分数
        self.gate = nn.Linear(d_model, num_experts)
        
    def forward(self, x):
        """
        前向传播
        参数:
            x: [batch_size, seq_len, d_model]
        返回:
            gates: 专家权重 [batch_size, seq_len, top_k]
            indices: 选中的专家索引 [batch_size, seq_len, top_k]
        """
        # 计算每个专家的logits
        logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        
        # 选择top-k个专家
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        
        # 对选中的专家做softmax归一化
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        return top_k_gates, top_k_indices


class MixtureOfExperts(nn.Module):
    """混合专家层（MoE Layer）- 替代标准FFN"""
    
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2, dropout=0.1):
        """
        参数:
            d_model: 模型维度
            d_ff: 专家网络隐藏层维度
            num_experts: 专家总数
            top_k: 每个token激活的专家数量（通常为2）
            dropout: Dropout比率
        """
        super(MixtureOfExperts, self).__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        # 创建多个专家网络
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) 
            for _ in range(num_experts)
        ])
        
        # 路由网络
        self.router = Router(d_model, num_experts, top_k)
        
    def forward(self, x):
        """
        前向传播
        参数:
            x: [batch_size, seq_len, d_model]
        返回:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. 路由：选择专家
        gates, indices = self.router(x)  # gates: [B, S, K], indices: [B, S, K]
        
        # 2. 初始化输出
        output = torch.zeros_like(x)  # [batch_size, seq_len, d_model]
        
        # 3. 对每个选中的专家进行计算
        # 方法1：循环方式（简单但效率较低）
        for i in range(self.top_k):
            # 获取第i个专家的索引和权重
            expert_indices = indices[:, :, i]  # [batch_size, seq_len]
            expert_gates = gates[:, :, i].unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # 对每个专家ID进行处理
            for expert_id in range(self.num_experts):
                # 找到使用当前专家的token
                mask = (expert_indices == expert_id)  # [batch_size, seq_len]
                
                if mask.any():
                    # 提取需要该专家处理的token
                    expert_input = x[mask]  # [num_tokens, d_model]
                    
                    # 专家处理
                    expert_output = self.experts[expert_id](expert_input)  # [num_tokens, d_model]
                    
                    # 将结果加权后加回到输出
                    # 注意：需要将mask扩展到d_model维度
                    mask_expanded = mask.unsqueeze(-1).expand_as(x)  # [batch_size, seq_len, d_model]
                    gate_expanded = expert_gates.expand_as(x)  # [batch_size, seq_len, d_model]
                    
                    # 创建临时输出张量
                    temp_output = torch.zeros_like(x)
                    temp_output[mask] = expert_output
                    
                    # 加权累加
                    output = output + temp_output * gate_expanded * mask_expanded.float()
        
        return output


class MoEEfficient(nn.Module):
    """高效的混合专家层实现（使用batch处理）"""
    
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2, dropout=0.1):
        """
        参数:
            d_model: 模型维度
            d_ff: 专家网络隐藏层维度
            num_experts: 专家总数
            top_k: 每个token激活的专家数量
            dropout: Dropout比率
        """
        super(MoEEfficient, self).__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        # 创建专家网络
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) 
            for _ in range(num_experts)
        ])
        
        # 路由网络
        self.router = Router(d_model, num_experts, top_k)
        
    def forward(self, x):
        """
        高效的前向传播实现
        参数:
            x: [batch_size, seq_len, d_model]
        """
        original_shape = x.shape
        batch_size, seq_len, d_model = x.shape
        
        # 1. 展平输入便于处理
        x_flat = x.view(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # 2. 路由
        gates, indices = self.router(x)  # [B, S, K]
        gates_flat = gates.view(-1, self.top_k)  # [B*S, K]
        indices_flat = indices.view(-1, self.top_k)  # [B*S, K]
        
        # 3. 初始化输出
        output_flat = torch.zeros_like(x_flat)  # [B*S, d_model]
        
        # 4. 为每个专家批量处理
        for expert_id in range(self.num_experts):
            # 找到所有使用该专家的位置
            expert_mask = (indices_flat == expert_id)  # [B*S, K]
            
            # 获取使用该专家的token索引
            token_indices, k_indices = torch.where(expert_mask)
            
            if len(token_indices) > 0:
                # 提取输入
                expert_input = x_flat[token_indices]  # [num_tokens, d_model]
                
                # 专家处理
                expert_output = self.experts[expert_id](expert_input)  # [num_tokens, d_model]
                
                # 获取对应的门控权重
                expert_gates = gates_flat[token_indices, k_indices].unsqueeze(-1)  # [num_tokens, 1]
                
                # 加权累加到输出
                output_flat[token_indices] += expert_output * expert_gates
        
        # 5. 恢复原始形状
        output = output_flat.view(original_shape)
        
        return output


class LoadBalancingLoss(nn.Module):
    """负载均衡损失 - 鼓励专家被均匀使用"""
    
    def __init__(self, num_experts):
        """
        参数:
            num_experts: 专家总数
        """
        super(LoadBalancingLoss, self).__init__()
        self.num_experts = num_experts
        
    def forward(self, gates, indices):
        """
        计算负载均衡损失
        参数:
            gates: 门控权重 [batch_size, seq_len, top_k]
            indices: 专家索引 [batch_size, seq_len, top_k]
        返回:
            loss: 标量损失值
        """
        # 计算每个专家被选中的频率
        expert_counts = torch.zeros(self.num_experts, device=gates.device)
        expert_gates_sum = torch.zeros(self.num_experts, device=gates.device)
        
        batch_size, seq_len, top_k = gates.shape
        total_tokens = batch_size * seq_len
        
        # 统计每个专家的使用情况
        for i in range(self.num_experts):
            mask = (indices == i)  # [batch_size, seq_len, top_k]
            expert_counts[i] = mask.sum().float()
            expert_gates_sum[i] = (gates * mask.float()).sum()
        
        # 计算频率和权重的均值
        freq = expert_counts / (total_tokens * top_k)  # 选中频率
        gate_mean = expert_gates_sum / (total_tokens * top_k)  # 平均权重
        
        # 负载均衡损失：freq * gate_mean 的方差越小越好
        loss = self.num_experts * (freq * gate_mean).sum()
        
        return loss


# 使用示例和对比
if __name__ == "__main__":
    print("=" * 80)
    print("混合专家模型(MoE)测试")
    print("=" * 80)
    
    # 设置参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048
    num_experts = 8
    top_k = 2
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 1. 标准FFN
    print("\n1. 标准FFN")
    print("-" * 80)
    standard_ffn = Expert(d_model, d_ff)
    standard_params = sum(p.numel() for p in standard_ffn.parameters())
    print(f"参数量: {standard_params:,}")
    
    with torch.no_grad():
        output_standard = standard_ffn(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output_standard.shape}")
    
    # 2. 基础MoE实现
    print("\n2. 基础MoE实现")
    print("-" * 80)
    moe = MixtureOfExperts(d_model, d_ff, num_experts, top_k)
    moe_params = sum(p.numel() for p in moe.parameters())
    print(f"专家总数: {num_experts}")
    print(f"每次激活专家数: {top_k}")
    print(f"总参数量: {moe_params:,}")
    print(f"实际计算参数量: {moe_params / num_experts * top_k:,.0f} (约为标准FFN的 {top_k/num_experts*100:.1f}%)")
    
    with torch.no_grad():
        output_moe = moe(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output_moe.shape}")
    
    # 3. 高效MoE实现
    print("\n3. 高效MoE实现")
    print("-" * 80)
    moe_efficient = MoEEfficient(d_model, d_ff, num_experts, top_k)
    
    with torch.no_grad():
        output_moe_efficient = moe_efficient(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output_moe_efficient.shape}")
    
    # 4. 负载均衡损失
    print("\n4. 负载均衡损失测试")
    print("-" * 80)
    lb_loss = LoadBalancingLoss(num_experts)
    
    with torch.no_grad():
        gates, indices = moe.router(x)
        loss_value = lb_loss(gates, indices)
    
    print(f"门控权重形状: {gates.shape}")
    print(f"专家索引形状: {indices.shape}")
    print(f"负载均衡损失: {loss_value.item():.4f}")
    
    # 5. 参数对比
    print("\n5. 参数效率对比")
    print("-" * 80)
    print(f"{'模型':<20} {'总参数量':<15} {'计算参数量':<15} {'效率提升'}")
    print("-" * 80)
    print(f"{'标准FFN':<20} {standard_params:>12,}  {standard_params:>12,}  {'1.0x'}")
    actual_compute = moe_params / num_experts * top_k
    efficiency = standard_params / actual_compute
    print(f"{'MoE (8专家, top2)':<20} {moe_params:>12,}  {actual_compute:>12,.0f}  {efficiency:.1f}x")
    
    # 6. 路由分析
    print("\n6. 路由分析（查看专家选择分布）")
    print("-" * 80)
    with torch.no_grad():
        gates, indices = moe.router(x)
        
        # 统计每个专家被选中的次数
        expert_usage = torch.zeros(num_experts)
        for i in range(num_experts):
            expert_usage[i] = (indices == i).sum().item()
        
        total_selections = expert_usage.sum().item()
        print(f"总选择次数: {total_selections:.0f}")
        print(f"\n专家使用分布:")
        for i in range(num_experts):
            usage_pct = expert_usage[i] / total_selections * 100
            bar = '█' * int(usage_pct / 2)
            print(f"  专家 {i}: {expert_usage[i]:>3.0f} 次 ({usage_pct:>5.1f}%) {bar}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    # 7. 性能优势总结
    print("\n💡 MoE优势总结:")
    print("-" * 80)
    print(f"✅ 模型容量提升: {num_experts}x (拥有{num_experts}个专家)")
    print(f"✅ 计算量降低: {num_experts/top_k:.1f}x (每次只用{top_k}个专家)")
    print(f"✅ 参数效率: 在参数量增加{num_experts}倍的情况下,计算量仅增加{top_k}倍")
    print(f"✅ 专业化能力: 不同专家可以学习处理不同类型的token")
    print("=" * 80)
