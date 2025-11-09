# Hybrid A* Path Planner for CARLA Automated Parking

This project implements a Hybrid A* path planning algorithm for an autonomous parking scenario within the CARLA simulator. It includes vehicle control, parking spot detection using YOLOv8-OBB (Oriented Bounding Boxes), and a C++ extension for performance-critical components.

이 프로젝트는 CARLA 시뮬레이터 내의 자율 주차 시나리오를 위한 Hybrid A* 경로 계획 알고리즘을 구현합니다. YOLOv8-OBB(Oriented Bounding Boxes)를 이용한 주차 공간 감지, 차량 제어 및 성능이 중요한 구성 요소를 위한 C++ 확장을 포함합니다.

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
This project uses a C++ extension (`fastbridge`) for performance-critical calculations. You need to compile it manually by following these steps from the project root directory.

1.  **Configure CMake & Build the project:**
    The commands to build are in `build.txt`. Execute them from the project root directory.
    ```bash
    # Find pybind11 and configure the build
    PYBIND_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$PYBIND_DIR"

    # Compile the project
    cmake --build build -j"$(nproc)"
    ```

2.  **Copy the compiled library:**
    After the build is complete, a file named `fastbridge.so` will be created inside the `build` directory. Copy this file to the project's root directory.
    ```bash
    cp build/fastbridge.so .
    ```

### 한국어
이 프로젝트는 성능이 중요한 계산을 위해 C++ 확장(`fastbridge`)을 사용합니다. 프로젝트 루트 디렉토리에서 아래 단계에 따라 직접 컴파일해야 합니다.

1.  **CMake 설정 및 프로젝트 빌드:**
    빌드 명령어는 `build.txt` 파일에 있습니다. 프로젝트 루트 디렉토리에서 이 명령어들을 실행하세요.
    ```bash
    # pybind11 경로를 찾고 빌드를 설정합니다
    PYBIND_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$PYBIND_DIR"

    # 프로젝트를 컴파일합니다
    cmake --build build -j"$(nproc)"
    ```

2.  **컴파일된 라이브러리 복사:**
    빌드가 완료되면 `build` 디렉토리 내에 `fastbridge.so` 파일이 생성됩니다. 이 파일을 프로젝트의 루트 디렉토리로 복사하세요.
    ```bash
    cp build/fastbridge.so .
    ```

## How to Run

### English
1.  Start the CARLA simulator.
2.  Run the main parking script from the project root directory:
    ```bash
    python3 run_carla_parking.py
    ```

### 한국어
1.  CARLA 시뮬레이터를 시작합니다.
2.  프로젝트 루트 디렉토리에서 메인 주차 스크립트를 실행합니다:
    ```bash
    python3 run_carla_parking.py
    ```