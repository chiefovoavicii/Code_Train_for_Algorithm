"""
GRPO (Group Relative Policy Optimization) 最小实现
用于语言模型的强化学习微调，是PPO的改进版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict


class PolicyNetwork(nn.Module):
    """策略网络（语言模型）"""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        """
        参数:
            vocab_size: 词汇表大小
            embed_dim: 词嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
        """
        super(PolicyNetwork, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM作为语言模型骨干
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        
        # 输出层（预测下一个token的概率）
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def forward(self, x, hidden=None):
        """
        前向传播
        参数:
            x: 输入token序列 [batch_size, seq_len]
            hidden: LSTM隐藏状态
        返回:
            logits: 输出logits [batch_size, seq_len, vocab_size]
            hidden: 新的隐藏状态
        """
        # 词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # LSTM前向传播
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)
        
        # 输出层
        logits = self.fc_out(lstm_out)  # [batch_size, seq_len, vocab_size]
        
        return logits, hidden
    
    def get_log_probs(self, sequences):
        """
        计算序列的对数概率
        参数:
            sequences: token序列 [batch_size, seq_len]
        返回:
            log_probs: 每个token的对数概率 [batch_size, seq_len-1]
        """
        # 输入是sequences[:-1]，目标是sequences[1:]
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]
        
        # 前向传播
        logits, _ = self.forward(inputs)  # [batch_size, seq_len-1, vocab_size]
        
        # 计算对数概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 收集目标token的对数概率
        target_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=targets.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        return target_log_probs
    
    def generate(self, start_tokens, max_length=20, temperature=1.0):
        """
        生成文本序列
        参数:
            start_tokens: 起始token [batch_size, start_len]
            max_length: 最大生成长度
            temperature: 温度参数（控制随机性）
        返回:
            generated: 生成的完整序列
            log_probs: 生成过程中的对数概率
        """
        batch_size = start_tokens.size(0)
        generated = start_tokens.clone()
        log_probs_list = []
        hidden = None
        
        for _ in range(max_length):
            # 获取最后一个token的logits
            logits, hidden = self.forward(generated[:, -1:], hidden)
            logits = logits[:, -1, :] / temperature  # [batch_size, vocab_size]
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            next_token = dist.sample()  # [batch_size]
            
            # 记录对数概率
            log_prob = dist.log_prob(next_token)
            log_probs_list.append(log_prob)
            
            # 拼接到生成序列
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        
        log_probs = torch.stack(log_probs_list, dim=1)  # [batch_size, max_length]
        
        return generated, log_probs


class ValueNetwork(nn.Module):
    """价值网络（评估状态价值）"""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        """
        参数:
            vocab_size: 词汇表大小
            embed_dim: 词嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
        """
        super(ValueNetwork, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        
        # 输出层（预测价值）
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入token序列 [batch_size, seq_len]
        返回:
            values: 状态价值 [batch_size, seq_len]
        """
        # 词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim]
        
        # 输出价值
        values = self.fc_out(lstm_out).squeeze(-1)  # [batch_size, seq_len]
        
        return values


class RewardModel(nn.Module):
    """奖励模型（评估生成质量）"""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        """
        参数:
            vocab_size: 词汇表大小
            embed_dim: 词嵌入维度
            hidden_dim: 隐藏层维度
        """
        super(RewardModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, sequences):
        """
        计算序列的奖励分数
        参数:
            sequences: token序列 [batch_size, seq_len]
        返回:
            rewards: 奖励分数 [batch_size]
        """
        embedded = self.embedding(sequences)
        lstm_out, (h_n, _) = self.lstm(embedded)
        
        # 使用最后一个隐藏状态
        reward = self.fc(h_n[-1]).squeeze(-1)  # [batch_size]
        
        return reward


class GRPO:
    """
    Group Relative Policy Optimization
    相比PPO的改进：使用组内相对优势，而不是全局优势
    """
    
    def __init__(
        self,
        policy: PolicyNetwork,
        value_net: ValueNetwork,
        reward_model: RewardModel,
        lr_policy=1e-4,
        lr_value=1e-4,
        gamma=0.99,
        lambda_gae=0.95,
        epsilon_clip=0.2,
        num_epochs=4,
        group_size=4,
        kl_coef=0.1
    ):
        """
        参数:
            policy: 策略网络
            value_net: 价值网络
            reward_model: 奖励模型
            lr_policy: 策略网络学习率
            lr_value: 价值网络学习率
            gamma: 折扣因子
            lambda_gae: GAE参数
            epsilon_clip: PPO裁剪参数
            num_epochs: 每次更新的训练轮数
            group_size: 组大小（GRPO核心参数）
            kl_coef: KL散度惩罚系数
        """
        self.policy = policy
        self.value_net = value_net
        self.reward_model = reward_model
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr_policy)
        self.value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr_value)
        
        # 超参数
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.epsilon_clip = epsilon_clip
        self.num_epochs = num_epochs
        self.group_size = group_size
        self.kl_coef = kl_coef
        
        # 保存参考策略（用于KL散度计算）
        self.ref_policy = None
        
    def compute_gae(self, rewards, values, dones):
        """
        计算广义优势估计(Generalized Advantage Estimation)
        参数:
            rewards: 奖励 [batch_size, seq_len]
            values: 价值估计 [batch_size, seq_len]
            dones: 结束标志 [batch_size, seq_len]
        返回:
            advantages: 优势函数 [batch_size, seq_len]
            returns: 回报 [batch_size, seq_len]
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            
            # TD误差
            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]
            
            # GAE
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[:, t]) * gae
            advantages[:, t] = gae
            returns[:, t] = advantages[:, t] + values[:, t]
        
        return advantages, returns
    
    def compute_group_advantages(self, advantages, group_size):
        """
        计算组内相对优势（GRPO的核心创新）
        参数:
            advantages: 原始优势 [batch_size, seq_len]
            group_size: 组大小
        返回:
            group_advantages: 组内归一化的优势
        """
        batch_size, seq_len = advantages.shape
        
        # 将batch分组
        num_groups = batch_size // group_size
        advantages_reshaped = advantages[:num_groups * group_size].view(num_groups, group_size, seq_len)
        
        # 组内标准化（减去组均值，除以组标准差）
        group_mean = advantages_reshaped.mean(dim=1, keepdim=True)
        group_std = advantages_reshaped.std(dim=1, keepdim=True) + 1e-8
        
        group_advantages = (advantages_reshaped - group_mean) / group_std
        group_advantages = group_advantages.view(-1, seq_len)
        
        # 处理剩余样本（如果batch_size不能被group_size整除）
        if batch_size % group_size != 0:
            remaining = advantages[num_groups * group_size:]
            remaining_normalized = (remaining - remaining.mean()) / (remaining.std() + 1e-8)
            group_advantages = torch.cat([group_advantages, remaining_normalized], dim=0)
        
        return group_advantages
    
    def ppo_loss(self, old_log_probs, new_log_probs, advantages, epsilon):
        """
        计算PPO损失
        参数:
            old_log_probs: 旧策略的对数概率
            new_log_probs: 新策略的对数概率
            advantages: 优势函数
            epsilon: 裁剪参数
        返回:
            loss: PPO损失
        """
        # 计算概率比
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 未裁剪的目标
        surr1 = ratio * advantages
        
        # 裁剪的目标
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        
        # PPO损失（取最小值，即最保守的更新）
        loss = -torch.min(surr1, surr2).mean()
        
        return loss
    
    def compute_kl_divergence(self, sequences, old_log_probs):
        """
        计算KL散度（防止策略偏离太远）
        参数:
            sequences: token序列
            old_log_probs: 参考策略的对数概率
        返回:
            kl_div: KL散度
        """
        # 获取当前策略的对数概率
        new_log_probs = self.policy.get_log_probs(sequences)
        
        # KL散度: KL(old||new) = old_log_probs - new_log_probs
        kl_div = (old_log_probs - new_log_probs).mean()
        
        return kl_div
    
    def train_step(self, prompts, generated_sequences, rewards_scores):
        """
        执行一步GRPO训练
        参数:
            prompts: 提示序列 [batch_size, prompt_len]
            generated_sequences: 生成的完整序列 [batch_size, total_len]
            rewards_scores: 奖励分数 [batch_size]
        返回:
            metrics: 训练指标字典
        """
        batch_size = generated_sequences.size(0)
        seq_len = generated_sequences.size(1) - 1  # 减1因为要预测下一个token
        
        # 1. 计算旧策略的对数概率（用于PPO）
        with torch.no_grad():
            old_log_probs = self.policy.get_log_probs(generated_sequences)
        
        # 2. 计算价值估计
        with torch.no_grad():
            values = self.value_net(generated_sequences[:, :-1])
        
        # 3. 构建奖励（只在序列末尾给奖励）
        rewards = torch.zeros(batch_size, seq_len, device=generated_sequences.device)
        rewards[:, -1] = rewards_scores  # 最后一个位置获得奖励
        
        # 4. 计算GAE优势
        dones = torch.zeros(batch_size, seq_len, device=generated_sequences.device)
        dones[:, -1] = 1.0  # 序列结束
        
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # 5. 计算组内相对优势（GRPO核心）
        group_advantages = self.compute_group_advantages(advantages, self.group_size)
        
        # 6. 多轮更新
        metrics = {
            'policy_loss': 0,
            'value_loss': 0,
            'kl_div': 0,
            'entropy': 0
        }
        
        for epoch in range(self.num_epochs):
            # 更新策略网络
            self.policy_optimizer.zero_grad()
            
            # 获取新的对数概率
            new_log_probs = self.policy.get_log_probs(generated_sequences)
            
            # 计算PPO损失
            policy_loss = self.ppo_loss(
                old_log_probs.detach(),
                new_log_probs,
                group_advantages.detach(),
                self.epsilon_clip
            )
            
            # KL散度惩罚
            kl_div = self.compute_kl_divergence(generated_sequences, old_log_probs.detach())
            
            # 总损失
            total_policy_loss = policy_loss + self.kl_coef * kl_div
            
            total_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()
            
            # 更新价值网络
            self.value_optimizer.zero_grad()
            
            new_values = self.value_net(generated_sequences[:, :-1])
            value_loss = F.mse_loss(new_values, returns.detach())
            
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
            self.value_optimizer.step()
            
            # 记录指标
            metrics['policy_loss'] += policy_loss.item()
            metrics['value_loss'] += value_loss.item()
            metrics['kl_div'] += kl_div.item()
        
        # 平均指标
        for key in metrics:
            metrics[key] /= self.num_epochs
        
        return metrics
    
    def generate_and_train(self, prompts, max_length=20):
        """
        生成序列并训练
        参数:
            prompts: 提示序列 [batch_size, prompt_len]
            max_length: 最大生成长度
        返回:
            generated: 生成的序列
            metrics: 训练指标
        """
        # 1. 生成序列
        with torch.no_grad():
            generated, log_probs = self.policy.generate(prompts, max_length)
        
        # 2. 计算奖励
        with torch.no_grad():
            rewards = self.reward_model(generated)
        
        # 3. 训练
        metrics = self.train_step(prompts, generated, rewards)
        
        return generated, metrics


# 使用示例
if __name__ == "__main__":
    print("=" * 80)
    print("GRPO (Group Relative Policy Optimization) 测试")
    print("=" * 80)
    
    # 设置参数
    vocab_size = 1000
    embed_dim = 128
    hidden_dim = 256
    batch_size = 8
    prompt_len = 5
    max_gen_length = 10
    group_size = 4  # GRPO的组大小
    
    # 创建网络
    policy = PolicyNetwork(vocab_size, embed_dim, hidden_dim, num_layers=2)
    value_net = ValueNetwork(vocab_size, embed_dim, hidden_dim, num_layers=2)
    reward_model = RewardModel(vocab_size, embed_dim, hidden_dim)
    
    # 创建GRPO训练器
    grpo = GRPO(
        policy=policy,
        value_net=value_net,
        reward_model=reward_model,
        lr_policy=1e-4,
        lr_value=1e-4,
        gamma=0.99,
        lambda_gae=0.95,
        epsilon_clip=0.2,
        num_epochs=4,
        group_size=group_size,
        kl_coef=0.1
    )
    
    print("\n1. 模型参数统计")
    print("-" * 80)
    policy_params = sum(p.numel() for p in policy.parameters())
    value_params = sum(p.numel() for p in value_net.parameters())
    reward_params = sum(p.numel() for p in reward_model.parameters())
    
    print(f"策略网络参数: {policy_params:,}")
    print(f"价值网络参数: {value_params:,}")
    print(f"奖励模型参数: {reward_params:,}")
    print(f"总参数: {policy_params + value_params + reward_params:,}")
    
    # 创建示例提示
    prompts = torch.randint(0, vocab_size, (batch_size, prompt_len))
    
    print("\n2. 生成测试")
    print("-" * 80)
    with torch.no_grad():
        generated, log_probs = policy.generate(prompts, max_length=max_gen_length)
    
    print(f"提示形状: {prompts.shape}")
    print(f"生成序列形状: {generated.shape}")
    print(f"对数概率形状: {log_probs.shape}")
    
    print("\n3. 奖励计算测试")
    print("-" * 80)
    with torch.no_grad():
        rewards = reward_model(generated)
    
    print(f"奖励形状: {rewards.shape}")
    print(f"奖励统计: 均值={rewards.mean().item():.4f}, 标准差={rewards.std().item():.4f}")
    
    print("\n4. GRPO训练测试")
    print("-" * 80)
    
    # 执行一次训练迭代
    generated, metrics = grpo.generate_and_train(prompts, max_length=max_gen_length)
    
    print(f"策略损失: {metrics['policy_loss']:.4f}")
    print(f"价值损失: {metrics['value_loss']:.4f}")
    print(f"KL散度: {metrics['kl_div']:.4f}")
    
    print("\n5. 组内优势计算测试")
    print("-" * 80)
    
    # 创建测试优势
    test_advantages = torch.randn(batch_size, max_gen_length)
    group_advantages = grpo.compute_group_advantages(test_advantages, group_size)
    
    print(f"原始优势形状: {test_advantages.shape}")
    print(f"组内优势形状: {group_advantages.shape}")
    
    # 验证组内归一化
    num_groups = batch_size // group_size
    for g in range(num_groups):
        group_advs = group_advantages[g * group_size:(g + 1) * group_size]
        print(f"组{g}: 均值={group_advs.mean().item():.4f}, 标准差={group_advs.std().item():.4f}")
    
    print("\n6. 完整训练循环示例")
    print("-" * 80)
    print("""
    # GRPO训练循环
    
    for iteration in range(num_iterations):
        # 1. 采样提示
        prompts = sample_prompts(batch_size)
        
        # 2. 生成并训练
        generated, metrics = grpo.generate_and_train(prompts, max_length=20)
        
        # 3. 记录指标
        print(f"Iter {iteration}: "
              f"Policy Loss={metrics['policy_loss']:.4f}, "
              f"Value Loss={metrics['value_loss']:.4f}, "
              f"KL={metrics['kl_div']:.4f}")
        
        # 4. 定期评估
        if iteration % 100 == 0:
            evaluate_model(policy, test_prompts)
    """)
    
    print("\n" + "=" * 80)
    print("💡 GRPO vs PPO 的关键区别")
    print("=" * 80)
    print("✅ PPO: 使用全局优势估计（所有样本的均值和标准差）")
    print("✅ GRPO: 使用组内相对优势（每组内部归一化）")
    print("✅ 优势: 减少不同批次之间的方差，训练更稳定")
    print("✅ 适用: 特别适合语言模型的RLHF（人类反馈强化学习）")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("📚 GRPO应用场景")
    print("=" * 80)
    print("1. 对话模型优化（ChatGPT风格）")
    print("2. 代码生成模型微调")
    print("3. 摘要生成优化")
    print("4. 任何需要RLHF的语言生成任务")
    print("=" * 80)
