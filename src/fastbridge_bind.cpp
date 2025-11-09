#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "fastbridge.hpp"

namespace py = pybind11;
using namespace fb;

static py::array_t<float> dijkstra_wrap(
    int gx, int gy,
    py::array_t<float, py::array::c_style | py::array::forcecast> ox,
    py::array_t<float, py::array::c_style | py::array::forcecast> oy,
    float reso, float robot_r,
    py::dict out_meta)
{
    std::vector<float> vox(ox.size()), voy(oy.size());
    std::memcpy(vox.data(), ox.data(), ox.nbytes());
    std::memcpy(voy.data(), oy.data(), oy.nbytes());

    Grid G;
    auto dist = dijkstra_holonomic(gx, gy, vox, voy, reso, robot_r, G);

    // 메타 반환: 격자 경계 등
    out_meta["minx"] = G.minx; out_meta["miny"] = G.miny;
    out_meta["maxx"] = G.maxx; out_meta["maxy"] = G.maxy;
    out_meta["xw"]   = G.xw;   out_meta["yw"]   = G.yw;
    out_meta["reso"] = G.reso;

    // (xw*yw) 1D 배열을 파이썬에 float32로 넘김
    return py::array_t<float>(
        {G.xw, G.yw},                             // shape
        {sizeof(float)*G.yw, sizeof(float)},      // strides
        dist.data()
    ).attr("copy")(); // 안전하게 복사본 반환
}

PYBIND11_MODULE(fastbridge, m){
    m.doc() = "Fast CPU bridges for holonomic heuristic & collision";
    m.def("holonomic_dijkstra", &dijkstra_wrap,
          py::arg("gx"), py::arg("gy"),
          py::arg("ox"), py::arg("oy"),
          py::arg("reso"), py::arg("robot_r"),
          py::arg("out_meta"),
          "Compute holonomic heuristic map (Dijkstra) on grid.");
    m.def("path_collision_any", [](py::array_t<float> xs,
                                   py::array_t<float> ys,
                                   py::array_t<float> yaws,
                                   py::array_t<float> ox,
                                   py::array_t<float> oy,
                                   float rf, float rb, float half_w){
        std::vector<float> vxs(xs.size()), vys(ys.size()), vyaws(yaws.size()),
                           vox(ox.size()), voy(oy.size());
        std::memcpy(vxs.data(), xs.data(), xs.nbytes());
        std::memcpy(vys.data(), ys.data(), ys.nbytes());
        std::memcpy(vyaws.data(), yaws.data(), yaws.nbytes());
        std::memcpy(vox.data(), ox.data(), ox.nbytes());
        std::memcpy(voy.data(), oy.data(), oy.nbytes());
        return (bool)path_collision_any(vxs,vys,vyaws,vox,voy,rf,rb,half_w);
    }, py::arg("xs"), py::arg("ys"), py::arg("yaws"),
       py::arg("ox"), py::arg("oy"),
       py::arg("rf"), py::arg("rb"), py::arg("half_w"));
}
