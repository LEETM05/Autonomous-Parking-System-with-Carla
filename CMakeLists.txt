cmake_minimum_required(VERSION 3.18)
project(fastbridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
# pybind11이 pip로 설치돼 있다면 자동 탐색이 잘 되는 편이지만,
# 안 잡히면 -DPYBIND11_DIR="/path/to/pybind11/share/cmake/pybind11" 를 추가로 넘겨줘.
find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND)
  message(STATUS "pybind11 CONFIG not found; trying module mode")
  find_package(pybind11 REQUIRED)
endif()

add_library(fastbridge MODULE
  src/fastbridge_bind.cpp
  src/fastbridge_cpu.cpp
)
target_include_directories(fastbridge PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(fastbridge PRIVATE pybind11::module)

# 파이썬 모듈 접두/접미 제거
set_target_properties(fastbridge PROPERTIES
  PREFIX ""
  SUFFIX ".so"
)
