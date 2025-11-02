# Google Colab 설정 가이드

이 문서는 강의 자료를 Google Colab에서 사용할 수 있도록 설정하는 방법을 설명합니다.

## 📋 단계별 가이드

### 1단계: GitHub에 코드 업로드

```bash
# Git 저장소 초기화 (아직 안 했다면)
git init

# 파일 추가
git add .
git commit -m "CIFAR-10 MLP 실습 코드 추가"

# GitHub 저장소 생성 후 (github.com에서)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

### 2단계: Colab 배지 링크 수정

**README.md**와 **노트북 파일**에서 다음 부분을 수정하세요:

**현재 설정:**
```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juho127/ClassificationTest/blob/main/cifar10_mlp_tutorial.ipynb)
```

✅ **이미 설정이 완료되었습니다!**
- GitHub 사용자명: `juho127`
- 저장소 이름: `ClassificationTest`

### 3단계: 학생들에게 공유

이제 학생들에게 다음 중 하나를 공유하세요:

1. **GitHub 저장소 링크**
   - `https://github.com/juho127/ClassificationTest`
   - 학생들이 README의 Colab 배지를 클릭

2. **직접 Colab 링크**
   - `https://colab.research.google.com/github/juho127/ClassificationTest/blob/main/cifar10_mlp_tutorial.ipynb`
   - 학생들이 링크를 클릭하면 바로 Colab에서 열림

## 🎓 학생 사용 방법

### 학생 입장에서의 사용법

1. **Colab 배지 클릭** 또는 직접 링크 접속
2. 구글 계정으로 로그인
3. **중요**: 런타임 > 런타임 유형 변경 > **GPU** 선택
4. 셀을 순서대로 실행 (Shift + Enter)

### 학생들에게 알려줄 팁

```
📌 강의 실습 안내

1. 위의 "Open in Colab" 버튼을 클릭하세요
2. 로그인 후 "런타임 > 런타임 유형 변경 > GPU"를 선택하세요
3. 코드 셀을 위에서부터 차례로 실행하세요 (Shift + Enter)
4. 자신의 Google Drive에 복사하려면: 파일 > 드라이브에 사본 저장

주의사항:
- Colab 세션은 12시간 후 또는 90분 미사용 시 종료됩니다
- 파일을 저장하려면 반드시 "드라이브에 사본 저장"을 하세요
```

## 🔧 추가 설정 (선택사항)

### Google Drive 연동

학생들이 결과를 저장하고 싶다면, 노트북에 다음 셀을 추가할 수 있습니다:

```python
# Google Drive 마운트 (선택사항)
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✓ Google Drive가 연결되었습니다.")
    print("📁 파일 저장 경로: /content/drive/MyDrive/")
```

### 데이터셋 캐싱

CIFAR-10을 매번 다운로드하지 않으려면:

```python
# Google Drive에 데이터셋 저장
if IN_COLAB:
    import os
    data_dir = '/content/drive/MyDrive/cifar10_data'
    os.makedirs(data_dir, exist_ok=True)
else:
    data_dir = './data'
```

## 📊 Colab vs 로컬 비교

| 항목 | Google Colab | 로컬 환경 |
|------|--------------|----------|
| 설치 | 불필요 | 필요 (Python, PyTorch 등) |
| GPU | 무료 제공 (T4) | 개인 GPU 필요 |
| 세션 | 12시간 제한 | 제한 없음 |
| 저장 | Drive 연동 필요 | 자동 저장 |
| 인터넷 | 필수 | 선택 |

## ❓ 자주 묻는 질문

### Q1: 배지를 클릭했는데 404 에러가 나요
A: GitHub 저장소 링크를 올바르게 수정했는지 확인하세요.

### Q2: GPU가 할당되지 않아요
A: 런타임 > 런타임 유형 변경 > GPU를 다시 선택해보세요. 간혹 사용량이 많으면 GPU를 할당받지 못할 수 있습니다.

### Q3: 세션이 끊겼어요
A: Colab은 90분 미사용 시 자동 종료됩니다. 주기적으로 셀을 실행하거나, 중요한 결과는 Drive에 저장하세요.

### Q4: 학생들이 코드를 수정하고 저장하려면?
A: "파일 > 드라이브에 사본 저장"을 클릭하면 개인 Drive에 복사본이 생성됩니다.

## 🎯 강의 활용 팁

1. **사전 준비**: 첫 수업 전에 학생들에게 구글 계정 준비를 안내하세요
2. **GPU 설정**: 첫 실습 시간에 GPU 설정 방법을 함께 실습하세요
3. **복사본 생성**: 학생들에게 자신의 Drive에 복사본을 만들도록 안내하세요
4. **결과 공유**: 학생들이 학습 결과 그래프를 스크린샷으로 제출하도록 할 수 있습니다

## 📞 문제 해결

문제가 발생하면:
1. 런타임 재시작: 런타임 > 런타임 다시 시작
2. 런타임 초기화: 런타임 > 런타임 초기화 및 다시 실행
3. 새 노트북: 파일 > 드라이브에 사본 저장으로 새 복사본 생성

