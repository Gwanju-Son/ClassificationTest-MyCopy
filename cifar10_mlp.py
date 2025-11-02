"""
CIFAR-10 데이터셋 MLP 분류 실습 코드
강의 실습용으로 작성된 코드입니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# 하이퍼파라미터 설정
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 클래스 이름
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (다층 퍼셉트론) 모델
    CIFAR-10 이미지를 분류하는 간단한 신경망
    """
    def __init__(self, input_size=3072, hidden_size1=512, hidden_size2=256, num_classes=10):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_size2, num_classes)
    
    def forward(self, x):
        # 이미지를 1차원으로 펼치기 (flatten)
        x = x.view(x.size(0), -1)
        
        # 첫 번째 은닉층
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # 두 번째 은닉층
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # 출력층
        x = self.fc3(x)
        return x


def load_data():
    """CIFAR-10 데이터셋 로드"""
    print("데이터셋을 불러오는 중...")
    
    # 데이터 전처리: 텐서로 변환하고 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 훈련 데이터셋
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 테스트 데이터셋
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    print(f"훈련 데이터: {len(train_dataset)}개")
    print(f"테스트 데이터: {len(test_dataset)}개")
    
    return train_loader, test_loader


def visualize_samples(loader):
    """데이터셋 샘플 시각화"""
    # 배치 하나 가져오기
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    # 이미지 정규화 해제
    images = images / 2 + 0.5
    
    # 그리드로 시각화
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('CIFAR-10 샘플 이미지', fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            # CHW -> HWC 형태로 변환
            img = images[idx].numpy().transpose((1, 2, 0))
            ax.imshow(img)
            ax.set_title(CLASSES[labels[idx]])
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('cifar10_samples.png', dpi=150, bbox_inches='tight')
    print("샘플 이미지가 'cifar10_samples.png'로 저장되었습니다.")
    plt.close()


def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """1 에포크 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    for images, labels in pbar:
        # 데이터를 디바이스로 이동
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 진행 상황 업데이트
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion):
    """모델 평가"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # 클래스별 정확도 계산을 위한 변수
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='평가 중'):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 클래스별 정확도 계산
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    # 클래스별 정확도 출력
    print("\n클래스별 정확도:")
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i]
        print(f'  {CLASSES[i]:10s}: {acc:.2f}%')
    
    return test_loss, test_acc


def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """학습 과정 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 손실 그래프
    ax1.plot(epochs, train_losses, 'b-', label='훈련 손실', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='테스트 손실', linewidth=2)
    ax1.set_xlabel('에포크', fontsize=12)
    ax1.set_ylabel('손실', fontsize=12)
    ax1.set_title('학습 과정: 손실', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 정확도 그래프
    ax2.plot(epochs, train_accs, 'b-', label='훈련 정확도', linewidth=2)
    ax2.plot(epochs, test_accs, 'r-', label='테스트 정확도', linewidth=2)
    ax2.set_xlabel('에포크', fontsize=12)
    ax2.set_ylabel('정확도 (%)', fontsize=12)
    ax2.set_title('학습 과정: 정확도', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("학습 과정 그래프가 'training_history.png'로 저장되었습니다.")
    plt.close()


def main():
    """메인 실행 함수"""
    print("="*60)
    print("CIFAR-10 MLP 분류 실습")
    print("="*60)
    print(f"사용 디바이스: {DEVICE}\n")
    
    # 1. 데이터 로드
    train_loader, test_loader = load_data()
    
    # 2. 샘플 이미지 시각화
    visualize_samples(train_loader)
    
    # 3. 모델 생성
    print("\n모델 생성 중...")
    model = MLP().to(DEVICE)
    print(model)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n전체 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    # 4. 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. 학습
    print(f"\n학습 시작 (총 {NUM_EPOCHS} 에포크)")
    print("-"*60)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        # 학습
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 평가
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  훈련 - 손실: {train_loss:.4f}, 정확도: {train_acc:.2f}%")
        print(f"  테스트 - 손실: {test_loss:.4f}, 정확도: {test_acc:.2f}%")
        
        # 최고 성능 모델 저장
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✓ 최고 성능 모델 저장됨 (정확도: {best_acc:.2f}%)")
        
        print("-"*60)
    
    # 6. 학습 과정 시각화
    print("\n학습 과정 시각화 중...")
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    # 7. 최종 결과
    print("\n" + "="*60)
    print("학습 완료!")
    print(f"최고 테스트 정확도: {best_acc:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()

