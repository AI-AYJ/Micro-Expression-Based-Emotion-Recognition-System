# Micro-Expression-Based-Emotion-Recognition-System

## 🧠 개요
이 프로젝트는 **얼굴 미세표정(Facial Micro-Expressions)** 을 분석하여 사람의 감정을 인식하는 딥러닝 기반 시스템입니다.  
업로드된 비디오나 웹캠 영상을 입력받아 초 단위로 감정 변화의 시점과 유형을 자동으로 탐지하며, 결과는 **Flask 기반 웹 인터페이스**를 통해 시각화됩니다.

## 🎯 목표
- 미세표정을 기반으로 감정을 분류하는 딥러닝 모델 구축
- 비디오 및 웹캠 입력을 통한 감정 예측 기능 구현
- 감정 변화 로그와 통계를 웹에서 실시간으로 제공

## 🧾 데이터셋
- **CASME II**
  - 26명의 피실험자, 247개 시퀀스, 200 fps
  - 감정 클래스: happiness, surprise, repression, disgust, sadness
  - 제거 클래스: others, fear (데이터 불균형 및 정의 모호성)
  - 주요 프레임: Onset, Apex, Offset

## ⚙️ 전처리
- Others, Fear 클래스 제거
- Apex Frame 추출
- Onset → Apex 구간 Optical Flow 계산
- 데이터 증강: Random Affine 변환, 수평 뒤집기 등

## 🧠 모델 구성
- Dual CNN 구조:  
  - Apex Frame용 CNN  
  - Optical Flow용 CNN
- **Softmax Score Ensemble**을 통해 최종 결과 도출
- 최종 정확도: **95.42%**

### 📌 적용 기법
- SimpleCNN 기반
- 히스토그램 평활화 및 CLAHE 실험 (과적합 문제로 미적용)
- CBAM (성능 저하로 최종 제외)

## 🌐 웹 구현 (Flask 기반)
### 📹 비디오 분석
- 대표 프레임 추출
- 감정 변화 통계 제공:
  - 총 감정 변화 횟수
  - 미세표정 발생 횟수
  - 감정 변화 로그 (이모지, 지속 시간 등)

### 🎥 웹캠 실시간 분석
- 0.4초 단위로 감정 예측
- 실시간 화면 출력 및 로그 저장

## 💡 활용 방안
- **심리 상담 및 임상 진단**: 진짜 감정 탐지, 치료 반응 모니터링
- **채용 및 면접 분석**: 지원자의 억제 감정 파악
- **온라인 교육**: 수강생의 집중도 및 이해도 추적
- **보안 및 감시**: 숨겨진 감정의 시각적 단서 제공
