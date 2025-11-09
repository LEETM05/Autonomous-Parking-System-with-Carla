# Hybrid A* Path Planner for CARLA Automated Parking

This project implements a Hybrid A* path planning algorithm for an autonomous parking scenario within the CARLA simulator. It includes vehicle control, parking spot detection using YOLO, and a C++ extension for performance-critical components.

이 프로젝트는 CARLA 시뮬레이터 내의 자율 주차 시나리오를 위한 Hybrid A* 경로 계획 알고리즘을 구현합니다. YOLO를 이용한 주차 공간 감지, 차량 제어 및 성능이 중요한 구성 요소를 위한 C++ 확장을 포함합니다.

## Requirements

### English
- **Python 3.8**
- **CARLA Simulator 0.9.13**
- Other dependencies are listed in `requirements.txt`.

### 한국어
- **Python 3.8**
- **CARLA Simulator 0.9.13**
- 그 외 의존성은 `requirements.txt`에 명시되어 있습니다.

## Installation

### English
1.  **Install CARLA 0.9.13**
    - Download and install CARLA 0.9.13 for your operating system (Linux or Windows).
    - Follow the official installation guide: [CARLA 0.9.13 Documentation](https://carla.readthedocs.io/en/0.9.13/)
    - Make sure to download the additional maps and place them in the appropriate directory as instructed.

2.  **Install Python Dependencies**
    - It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

### 한국어
1.  **CARLA 0.9.13 설치**
    - 사용 중인 운영체제(Linux 또는 Windows)에 맞는 CARLA 0.9.13을 다운로드하여 설치합니다.
    - 공식 설치 가이드를 따르세요: [CARLA 0.9.13 공식 문서](https://carla.readthedocs.io/en/0.9.13/)
    - 안내에 따라 추가 맵을 다운로드하고 올바른 디렉토리에 배치해야 합니다.

2.  **Python 의존성 설치**
    - 가상 환경 사용을 권장합니다.
    ```bash
    pip install -r requirements.txt
    ```

## Build

### English
This project uses a C++ extension for performance-critical calculations, which needs to be compiled. The build script `build.txt` handles the process.

Run the following command from the project root directory:
```bash
bash build.txt
```
This will create the compiled library (`fastbridge.so`) inside the `build` directory and copy it to the root directory.

### 한국어
이 프로젝트는 성능이 중요한 계산을 위해 C++ 확장을 사용하므로 컴파일이 필요합니다. `build.txt` 빌드 스크립트가 이 과정을 처리합니다.

프로젝트 루트 디렉토리에서 다음 명령어를 실행하세요:
```bash
bash build.txt
```
이 명령어는 컴파일된 라이브러리(`fastbridge.so`)를 `build` 디렉토리에 생성하고, 루트 디렉토리로 복사합니다.

## How to Run

### English
1.  Start the CARLA simulator.
2.  Run the main parking script from the project root directory:
    ```bash
    python3.8 run_carla_parking.py
    ```

### 한국어
1.  CARLA 시뮬레이터를 시작합니다.
2.  프로젝트 루트 디렉토리에서 메인 주차 스크립트를 실행합니다:
    ```bash
    python3.8 run_carla_parking.py
    ```