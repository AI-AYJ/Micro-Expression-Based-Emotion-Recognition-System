# Micro-Expression-Based-Emotion-Recognition-System

## 개요
이 프로젝트는 **얼굴 미세표정(Facial Micro-Expressions)** 을 분석하여 사람의 감정을 인식하는 딥러닝 기반 시스템입니다.
미세표정은 보통 0.2초에서 0.5초 사이에 나타나는 무의식적이고 매우 짧은 표정으로, 개인이 감정을 숨기려 할 때 순간적으로 드러나는 진짜 감정을 반영합니다.
업로드된 비디오나 웹캠 영상을 입력받아 초 단위로 감정 변화의 시점과 유형을 자동으로 탐지하며, 결과는 **Flask 기반 웹 인터페이스**를 통해 시각화됩니다.


## 목표
- 미세표정을 기반으로 감정을 분류하는 딥러닝 모델 구축
- 비디오 및 웹캠 입력을 통한 감정 예측 기능 구현
- 감정 변화 로그와 통계를 웹에서 실시간으로 제공

<img width="863" alt="image" src="https://github.com/user-attachments/assets/fd9aa68c-2066-4422-aaf1-670a94ea7b31" />

## 데이터셋
- **CASME II**
  - 26명의 피실험자, 247개 시퀀스, 200 fps
  - 감정 클래스: happiness, surprise, repression, disgust, sadness
  - 제거 클래스: others, fear (데이터 불균형 및 정의 모호성)
  - 주요 프레임: Onset (미세표정이 시작되는 프레임), Apex (미세표정이 가장 강한 프레임), Offset (미세표정이 끝나는 프레임)

## 전처리
- Others, Fear 클래스 제거 (Others:애매한 클래스, Fear:극소수의 데이터 → 불균형이 일어날 수 있음)
 <img src="https://github.com/user-attachments/assets/ad4e7794-96a7-4d07-8bbe-db72245bdd6d" width="400" />

- Apex Frame 추출
- Onset → Apex 구간 Optical Flow 계산
- 데이터 증강: HorizontalFlip: Flip(p=0.3) + RandomAffine: translate = (0.02, 0.02)
- 
## 모델 구성
- Dual CNN 구조:  
  - Apex Frame용 CNN  
  - Optical Flow용 CNN
   
  <br><br>
  <img src="https://github.com/user-attachments/assets/1f27533d-d70b-40a8-ba42-1d1b65f9b59f" width="300" hspace="20" />
  <img src="https://github.com/user-attachments/assets/9f2ee7d7-d6ba-4099-aa46-60ec47e91dec" width="300" />

- **Softmax Score Ensemble**을 통해 최종 결과 도출
- 최종 정확도: **95.42%**
<img src="https://github.com/user-attachments/assets/b89030be-8406-4309-a901-11600125ab00" width="400">

### 적용 기법
- SimpleCNN 기반
- 히스토그램 평활화 및 CLAHE 실험 (과적합 문제로 미적용)
- CBAM (성능 저하로 최종 제외)

## 웹 구현 (Flask 기반)

**Flask 코드는 myproject폴더 안에
![image](https://github.com/user-attachments/assets/199fb914-3091-405b-a086-defcd2caa359)


### 비디오 분석
- 대표 프레임 추출
- 감정 변화 통계 제공:
  - 총 감정 변화 횟수
  - 미세표정 발생 횟수
  - 감정 변화 로그 (이모지, 지속 시간 등)

### 웹캠 실시간 분석
- 0.4초 단위로 감정 예측
- 실시간 화면 출력 및 로그 저장

## 활용 방안
- **심리 상담 및 임상 진단**: 진짜 감정 탐지, 치료 반응 모니터링
- **채용 및 면접 분석**: 지원자의 억제 감정 파악
- **온라인 교육**: 수강생의 집중도 및 이해도 추적
- **보안 및 감시**: 숨겨진 감정의 시각적 단서 제공

----------------------------------------------------------

## 느낀점
- CASME2만 사용했기 때문에 데이터 수가 적어서 좀 더 다양한 시도를 못한 점이 아쉽습니다. 또한 fear를 불균형 때문에 제거했지만 sadness클래스도 수가 충분하지 않아서 클래스 불균형이 일어났었고 그 점을 보완하기 위한 방안을 적용하지 못한 점이 추후 해결해야할 문제라고 생각합니다.
