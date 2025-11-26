import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class Trainer:
    def __init__(self, model, train_loader, test_loader, optimizer, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(self.device)

    def fit(self, epochs=10):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
            self.evaluate()

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.test_loader)
        print(f"Test Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # 配置参数
    IN_DIM = 512
    HIDDEN_DIM = 256
    OUT_DIM = 1
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 生成模拟数据
    # 假设输入是 (N, 512)，输出是 (N, 1)
    x = torch.randn(1000, IN_DIM)
    y = torch.randn(1000, OUT_DIM)

    # 2. 划分数据集
    train_size = int(0.8 * len(x))
    train_x, test_x = x[:train_size], x[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    # 3. 创建 DataLoader
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 初始化模型、优化器、损失函数
    model = MLP(IN_DIM, HIDDEN_DIM, OUT_DIM)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # 5. 初始化训练器并开始训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE
    )

    print(f"Start training on {DEVICE}...")
    trainer.fit(epochs=EPOCHS)
