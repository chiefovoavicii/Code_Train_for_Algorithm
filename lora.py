"""
LoRA微调最小实现
提供LoRA注入工具类，并展示如何在SFT与GRPO流程中复用LoRA仅训练少量参数
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from GRPO import GRPO, PolicyNetwork, ValueNetwork, RewardModel


@dataclass
class LoRAConfig:
    """LoRA配置项"""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Optional[List[str]] = field(default=None)
    exclude_modules: Optional[List[str]] = field(default=None)

    def should_replace(self, module_name: str) -> bool:
        """判断模块是否注入LoRA"""

        if self.exclude_modules and any(ex in module_name for ex in self.exclude_modules):
            return False
        if not self.target_modules:
            return True
        return any(t in module_name for t in self.target_modules)


class LoRALinear(nn.Module):
    """线性层LoRA适配器"""

    def __init__(self, linear: nn.Linear, config: LoRAConfig):
        super().__init__()
        if config.rank <= 0:
            raise ValueError("LoRA rank必须大于0")

        self.linear = linear
        self.rank = config.rank
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # 原始参数冻结
        for param in self.linear.parameters():
            param.requires_grad = False

        # LoRA新增两个低秩矩阵
        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.linear.out_features, self.rank))

        # 初始化：A用Kaiming，B用0有助于稳定
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)

        # LoRA增量：先乘A降维，再乘B升维
        lora_output = F.linear(self.dropout(x), self.lora_A)
        lora_output = F.linear(lora_output, self.lora_B) * self.scaling
        return base + lora_output


class LoRAAdapter:
    """对任意模型注入LoRA层的工具"""

    def __init__(self, model: nn.Module, config: LoRAConfig):
        self.model = model
        self.config = config
        self.lora_layers: List[LoRALinear] = []

    def inject(self) -> nn.Module:
        """递归遍历模型并替换目标线性层"""

        def _inject(module: nn.Module, prefix: str = "") -> None:
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Linear) and self.config.should_replace(full_name):
                    lora_layer = LoRALinear(child, self.config)
                    setattr(module, name, lora_layer)
                    self.lora_layers.append(lora_layer)
                else:
                    _inject(child, full_name)

        _inject(self.model)
        return self.model

    def parameters(self) -> Iterable[nn.Parameter]:
        for layer in self.lora_layers:
            for param in layer.parameters():
                if param.requires_grad:
                    yield param

    def to(self, device: torch.device) -> nn.Module:
        self.model.to(device)
        return self.model


class LoRASFTTrainer:
    """LoRA + SFT最小训练框架"""

    def __init__(self, model: nn.Module, config: LoRAConfig, lr: float = 1e-4, device: Optional[torch.device] = None):
        self.model = model
        self.adapter = LoRAAdapter(self.model, config)
        self.adapter.inject()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.adapter.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def train_step(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> float:
        self.model.train()

        if labels is None:
            inputs = input_ids[:, :-1].to(self.device)
            labels = input_ids[:, 1:].to(self.device)
        else:
            inputs = input_ids.to(self.device)
            labels = labels.to(self.device)

        logits, _ = self.model(inputs)
        vocab_size = logits.size(-1)

        loss = self.criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class LoRAGRPOTrainer:
    """将LoRA注入GRPO策略网络的包装类"""

    def __init__(self, grpo: GRPO, config: LoRAConfig, lr: float = 1e-4, train_value_lora: bool = False):
        self.grpo = grpo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 仅对策略网络做LoRA
        self.policy_adapter = LoRAAdapter(self.grpo.policy, config)
        self.policy_adapter.inject()
        self.grpo.policy.to(self.device)

        # 可选对价值网络也注入LoRA
        if train_value_lora:
            self.value_adapter = LoRAAdapter(self.grpo.value_net, config)
            self.value_adapter.inject()
            self.grpo.value_net.to(self.device)
        else:
            self.value_adapter = None
            self.grpo.value_net.to(self.device)

        # 只优化LoRA新增参数
        self.grpo.policy_optimizer = torch.optim.Adam(self.policy_adapter.parameters(), lr=lr)
        if self.value_adapter:
            self.grpo.value_optimizer = torch.optim.Adam(self.value_adapter.parameters(), lr=lr)

    def generate_and_train(self, prompts: torch.Tensor, max_length: int = 20):
        prompts = prompts.to(self.device)
        return self.grpo.generate_and_train(prompts, max_length=max_length)


if __name__ == "__main__":
    torch.manual_seed(42)

    vocab_size = 1000
    config = LoRAConfig(rank=4, alpha=16, dropout=0.05, target_modules=["linear", "fc_out"])

    # 1. SFT示例
    policy_sft = PolicyNetwork(vocab_size=vocab_size, embed_dim=128, hidden_dim=256, num_layers=2)
    sft_trainer = LoRASFTTrainer(policy_sft, config, lr=5e-4)

    sample_batch = torch.randint(0, vocab_size, (2, 16))
    sft_loss = sft_trainer.train_step(sample_batch)
    print(f"SFT单步Loss: {sft_loss:.4f}")

    # 2. GRPO示例
    policy_rl = PolicyNetwork(vocab_size=vocab_size, embed_dim=128, hidden_dim=256, num_layers=2)
    value_rl = ValueNetwork(vocab_size=vocab_size, embed_dim=128, hidden_dim=256, num_layers=2)
    reward_model = RewardModel(vocab_size=vocab_size, embed_dim=128, hidden_dim=256)

    grpo = GRPO(policy=policy_rl, value_net=value_rl, reward_model=reward_model, group_size=2)
    grpo_trainer = LoRAGRPOTrainer(grpo, config, lr=5e-4)

    prompts = torch.randint(0, vocab_size, (2, 8))
    generated, metrics = grpo_trainer.generate_and_train(prompts, max_length=6)

    print(f"生成序列形状: {generated.shape}")
    print(f"GRPO指标: {metrics}")
