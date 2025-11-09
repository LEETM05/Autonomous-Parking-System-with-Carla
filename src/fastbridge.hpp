#pragma once
#include <vector>
#include <cstdint>
#include <limits>
#include <queue>
#include <cmath>

namespace fb {

// 격자 파라미터
struct Grid {
    int minx, miny, maxx, maxy; // grid index 경계
    int xw, yw;                 // 폭
    float reso;                 // [m]
};

// 휴리스틱(Dijkstra) 결과: xw*yw float32 맵
std::vector<float> dijkstra_holonomic(
    int gx, int gy,                 // goal grid index
    const std::vector<float>& ox_m, // obstacles [m]
    const std::vector<float>& oy_m,
    float reso, float robot_r,      // grid reso, robot radius [m]
    Grid& outG                      // out: 격자 파라미터
);

// 충돌 여부(경로 시퀀스 한 번에 검사; 축 회전 직사각 차량 모델)
bool path_collision_any(
    const std::vector<float>& xs,
    const std::vector<float>& ys,
    const std::vector<float>& yaws,
    const std::vector<float>& ox,
    const std::vector<float>& oy,
    float rf, float rb, float half_w
);

} // namespace fb
