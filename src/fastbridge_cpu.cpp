#include "fastbridge.hpp"
#include <algorithm>

namespace fb {

// 8방향 이동(대각선 가중치 sqrt(2))
static const int DX[8] = {1,-1,0,0, 1, 1,-1,-1};
static const int DY[8] = {0, 0,1,-1, 1,-1, 1,-1};
static const float DC[8] = {1.f,1.f,1.f,1.f, std::sqrt(2.f),std::sqrt(2.f),std::sqrt(2.f),std::sqrt(2.f)};

static inline int toIndex(int x, int y, const Grid& G) {
    return (x - G.minx) * G.yw + (y - G.miny);
}

static Grid makeGrid(const std::vector<float>& ox, const std::vector<float>& oy, float reso, int pad=5) {
    float minx_m = *std::min_element(ox.begin(), ox.end());
    float maxx_m = *std::max_element(ox.begin(), ox.end());
    float miny_m = *std::min_element(oy.begin(), oy.end());
    float maxy_m = *std::max_element(oy.begin(), oy.end());
    int minx = static_cast<int>(std::floor(minx_m / reso)) - pad;
    int maxx = static_cast<int>(std::ceil (maxx_m / reso)) + pad;
    int miny = static_cast<int>(std::floor(miny_m / reso)) - pad;
    int maxy = static_cast<int>(std::ceil (maxy_m / reso)) + pad;
    Grid G;
    G.minx=minx; G.miny=miny; G.maxx=maxx; G.maxy=maxy;
    G.xw = maxx - minx; G.yw = maxy - miny; G.reso = reso;
    return G;
}

static std::vector<uint8_t> buildObsmap(
    const std::vector<float>& ox, const std::vector<float>& oy,
    float robot_r, const Grid& G)
{
    // 단순한 점-원 반경 버퍼로 맵 표시
    std::vector<uint8_t> occ(G.xw * G.yw, 0);
    const float rr = robot_r / G.reso; // 그리드 칸 단위
    const int rri = std::max(1, (int)std::ceil(rr));

    for (size_t i=0;i<ox.size();++i){
        int gx = (int)std::round(ox[i]/G.reso);
        int gy = (int)std::round(oy[i]/G.reso);
        for (int dx=-rri; dx<=rri; ++dx){
            for (int dy=-rri; dy<=rri; ++dy){
                int xx = gx + dx, yy = gy + dy;
                if (xx<=G.minx || xx>=G.maxx || yy<=G.miny || yy>=G.maxy) continue;
                int idx = toIndex(xx,yy,G);
                occ[idx] = 1;
            }
        }
    }
    return occ;
}

std::vector<float> dijkstra_holonomic(
    int gx, int gy,
    const std::vector<float>& ox_m,
    const std::vector<float>& oy_m,
    float reso, float robot_r,
    Grid& outG)
{
    Grid G = makeGrid(ox_m, oy_m, reso, /*pad=*/5);
    // goal이 그리드 밖이면 확장
    if (gx<=G.minx || gx>=G.maxx || gy<=G.miny || gy>=G.maxy){
        int pad = 5;
        G.minx = std::min(G.minx, gx-pad);
        G.miny = std::min(G.miny, gy-pad);
        G.maxx = std::max(G.maxx, gx+pad);
        G.maxy = std::max(G.maxy, gy+pad);
        G.xw = G.maxx - G.minx;
        G.yw = G.maxy - G.miny;
    }

    auto occ = buildObsmap(ox_m, oy_m, robot_r, G);

    const int N = G.xw*G.yw;
    std::vector<float> dist(N, std::numeric_limits<float>::infinity());
    std::vector<uint8_t> used(N, 0);

    auto inside = [&](int x,int y){
        return !(x<=G.minx || x>=G.maxx || y<=G.miny || y>=G.maxy);
    };

    struct QN{float c; int x,y;};
    struct Cmp{bool operator()(const QN&a, const QN&b)const{return a.c>b.c;}};
    std::priority_queue<QN,std::vector<QN>,Cmp> pq;

    int gidx = toIndex(gx, gy, G);
    dist[gidx] = 0.f;
    pq.push({0.f, gx, gy});

    while(!pq.empty()){
        auto [c,x,y] = pq.top(); pq.pop();
        int idx = toIndex(x,y,G);
        if (used[idx]) continue;
        used[idx] = 1;

        for (int k=0;k<8;++k){
            int nx=x+DX[k], ny=y+DY[k];
            if (!inside(nx,ny)) continue;
            int nidx = toIndex(nx,ny,G);
            if (occ[nidx]) continue;
            float nc = c + DC[k];
            if (nc < dist[nidx]){
                dist[nidx]=nc;
                pq.push({nc,nx,ny});
            }
        }
    }

    outG = G;
    return dist; // xw*yw length (열 우선: (x-minx)*yw + (y-miny))
}

// 차량을 길이(앞 RF, 뒤 RB), 폭(half_w)로 근사한 회전 직사각형 충돌
static inline bool rectHit(
    float cx, float cy, float yaw,
    float rf, float rb, float half_w,
    float ox, float oy)
{
    // 차량 좌표계로 변환
    float dx = ox - cx;
    float dy = oy - cy;
    float c = std::cos(yaw), s = std::sin(yaw);
    float lx =  c*dx + s*dy;
    float ly = -s*dx + c*dy;

    // 차체 경계
    float x_min = -rb, x_max = rf;
    float y_min = -half_w, y_max = half_w;

    return (lx > x_min && lx < x_max && ly > y_min && ly < y_max);
}

bool path_collision_any(
    const std::vector<float>& xs,
    const std::vector<float>& ys,
    const std::vector<float>& yaws,
    const std::vector<float>& ox,
    const std::vector<float>& oy,
    float rf, float rb, float half_w)
{
    const size_t T = xs.size();
    for (size_t t=0; t<T; ++t){
        float cx=xs[t], cy=ys[t], cyaw=yaws[t];
        // 근방만 거칠게 검사하려면 사전 셀/해시가 있으면 좋지만
        // 구현 단순화를 위해 여기선 전수검사(ox,oy가 다운샘플링 되어 있다고 가정)
        for (size_t i=0;i<ox.size();++i){
            if (rectHit(cx,cy,cyaw,rf,rb,half_w,ox[i],oy[i])) return true;
        }
    }
    return false;
}

} // namespace fb
