import heapq
import math
import numpy as np
import matplotlib.pyplot as plt

try:
    import fastbridge as _fb
except ImportError:
    _fg = None

class Node:
    def __init__(self, x, y, cost, pind):
        self.x = x  # x position of node
        self.y = y  # y position of node
        self.cost = cost  # g cost of node
        self.pind = pind  # parent index of node


class Para:
    def __init__(self, minx, miny, maxx, maxy, xw, yw, reso, motion):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.xw = xw
        self.yw = yw
        self.reso = reso  # resolution of grid world
        self.motion = motion  # motion set


def astar_planning(sx, sy, gx, gy, ox, oy, reso, rr):
    """
    return path of A*.
    :param sx: starting node x [m]
    :param sy: starting node y [m]
    :param gx: goal node x [m]
    :param gy: goal node y [m]
    :param ox: obstacles x positions [m]
    :param oy: obstacles y positions [m]
    :param reso: xy grid resolution
    :param rr: robot radius
    :return: path
    """

    n_start = Node(round(sx / reso), round(sy / reso), 0.0, -1)
    n_goal = Node(round(gx / reso), round(gy / reso), 0.0, -1)

    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]

    P, obsmap = calc_parameters(ox, oy, rr, reso)

    open_set, closed_set = dict(), dict()
    open_set[calc_index(n_start, P)] = n_start

    q_priority = []
    heapq.heappush(q_priority,
                   (fvalue(n_start, n_goal), calc_index(n_start, P)))

    while True:
        if not open_set:
            break

        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        for i in range(len(P.motion)):
            node = Node(n_curr.x + P.motion[i][0],
                        n_curr.y + P.motion[i][1],
                        n_curr.cost + u_cost(P.motion[i]), ind)

            if not check_node(node, P, obsmap):
                continue

            n_ind = calc_index(node, P)
            if n_ind not in closed_set:
                if n_ind in open_set:
                    if open_set[n_ind].cost > node.cost:
                        open_set[n_ind].cost = node.cost
                        open_set[n_ind].pind = ind
                else:
                    open_set[n_ind] = node
                    heapq.heappush(q_priority,
                                   (fvalue(node, n_goal), calc_index(node, P)))

    pathx, pathy = extract_path(closed_set, n_start, n_goal, P)

    return pathx, pathy

def calc_holonomic_heuristic_with_obstacle(node, ox, oy, reso, rr):
    # ---- CPP 변환
    if _fb is not None:
        n_goal = Node(round(node.x[-1] / reso), round(node.y[-1] / reso), 0.0, -1)
        meta = {}
        h_np = _fb.holonomic_dijkstra(
            int(n_goal.x), int(n_goal.y),
            np.asarray(ox, dtype=np.float32),
            np.asarray(oy, dtype=np.float32),
            float(reso), float(rr), meta
        )  # shape=(xw,yw), float32
        # 파이썬 쪽은 hmap[x][y] 인덱싱이라 list[list]로 변환:
        hmap = [[float(h_np[x, y]) for y in range(h_np.shape[1])]
                for x in range(h_np.shape[0])]
        return hmap


    n_goal = Node(round(node.x[-1] / reso), round(node.y[-1] / reso), 0.0, -1)

    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]

    P, obsmap = calc_parameters(ox, oy, reso, rr)

    open_set, closed_set = dict(), dict()
    open_set[calc_index(n_goal, P)] = n_goal

    q_priority = []
    heapq.heappush(q_priority, (n_goal.cost, calc_index(n_goal, P)))

    while True:
        if not open_set:
            break

        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        for i in range(len(P.motion)):
            node = Node(n_curr.x + P.motion[i][0],
                        n_curr.y + P.motion[i][1],
                        n_curr.cost + u_cost(P.motion[i]), ind)

            if not check_node(node, P, obsmap):
                continue

            n_ind = calc_index(node, P)
            if n_ind not in closed_set:
                if n_ind in open_set:
                    if open_set[n_ind].cost > node.cost:
                        open_set[n_ind].cost = node.cost
                        open_set[n_ind].pind = ind
                else:
                    open_set[n_ind] = node
                    heapq.heappush(q_priority, (node.cost, calc_index(node, P)))

    hmap = [[np.inf for _ in range(P.yw)] for _ in range(P.xw)]

    for n in closed_set.values():
        hmap[n.x - P.minx][n.y - P.miny] = n.cost

    return hmap


def check_node(node, P, obsmap):
    if node.x <= P.minx or node.x >= P.maxx or \
            node.y <= P.miny or node.y >= P.maxy:
        return False

    if obsmap[node.x - P.minx][node.y - P.miny]:
        return False

    return True


def u_cost(u):
    return math.hypot(u[0], u[1])


def fvalue(node, n_goal):
    return node.cost + h(node, n_goal)


def h(node, n_goal):
    return math.hypot(node.x - n_goal.x, node.y - n_goal.y)


def calc_index(node, P):
    return (node.y - P.miny) * P.xw + (node.x - P.minx)


def calc_parameters(ox, oy, rr, reso):
    minx, miny = round(min(ox)), round(min(oy))
    maxx, maxy = round(max(ox)), round(max(oy))
    xw, yw = maxx - minx, maxy - miny

    motion = get_motion()
    P = Para(minx, miny, maxx, maxy, xw, yw, reso, motion)
    obsmap = calc_obsmap(ox, oy, rr, P)

    return P, obsmap


def calc_obsmap(ox, oy, rr, P):
    obsmap = [[False for _ in range(P.yw)] for _ in range(P.xw)]

    for x in range(P.xw):
        xx = x + P.minx
        for y in range(P.yw):
            yy = y + P.miny
            for oxx, oyy in zip(ox, oy):
                if math.hypot(oxx - xx, oyy - yy) <= rr / P.reso:
                    obsmap[x][y] = True
                    break

    return obsmap


def extract_path(closed_set, n_start, n_goal, P):
    pathx, pathy = [n_goal.x], [n_goal.y]
    n_ind = calc_index(n_goal, P)

    while True:
        node = closed_set[n_ind]
        pathx.append(node.x)
        pathy.append(node.y)
        n_ind = node.pind

        if node == n_start:
            break

    pathx = [x * P.reso for x in reversed(pathx)]
    pathy = [y * P.reso for y in reversed(pathy)]

    return pathx, pathy


def get_motion():
    motion = [[-1, 0], [-1, 1], [0, 1], [1, 1],
              [1, 0], [1, -1], [0, -1], [-1, -1]]

    return motion


def get_env():
    ox, oy = [], []

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)

    return ox, oy

    # ======================= visualization =======================

# def dijkstra_planning(sx, sy, gx, gy, ox, oy, reso, rr):
#     """A*에서 휴리스틱을 0으로 둔 버전 (우선순위 = g)"""
#     n_start = Node(round(sx / reso), round(sy / reso), 0.0, -1)
#     n_goal  = Node(round(gx / reso), round(gy / reso), 0.0, -1)

#     ox = [x / reso for x in ox]; oy = [y / reso for y in oy]
#     P, obsmap = calc_parameters(ox, oy, rr, reso)

#     open_set, closed_set = {calc_index(n_start, P): n_start}, {}
#     pq = []  # (g, index)
#     heapq.heappush(pq, (n_start.cost, calc_index(n_start, P)))

#     while open_set:
#         _, ind = heapq.heappop(pq)
#         n_curr = open_set[ind]; closed_set[ind] = n_curr; open_set.pop(ind)

#         if n_curr.x == round(gx / reso) and n_curr.y == round(gy / reso):
#             break

#         for dx,dy in P.motion:
#             node = Node(n_curr.x + dx, n_curr.y + dy, n_curr.cost + u_cost([dx,dy]), ind)
#             if not check_node(node, P, obsmap): continue
#             n_ind = calc_index(node, P)
#             if n_ind in closed_set: continue
#             if n_ind not in open_set or open_set[n_ind].cost > node.cost:
#                 open_set[n_ind] = node
#                 heapq.heappush(pq, (node.cost, n_ind))

#     pathx, pathy = extract_path(closed_set, n_start, n_goal, P)
#     return pathx, pathy

def dijkstra_planning(sx, sy, gx, gy, ox, oy, reso, rr):
    n_start = Node(round(sx / reso), round(sy / reso), 0.0, -1)
    n_goal  = Node(round(gx / reso), round(gy / reso), 0.0, -1)

    # 격자/장애물 전개
    oxx = [x / reso for x in ox]; oyy = [y / reso for y in oy]
    P, obsmap = calc_parameters(oxx, oyy, rr, reso)

    open_set, closed_set = {calc_index(n_start, P): n_start}, {}
    import heapq
    pq = []
    heapq.heappush(pq, (n_start.cost, calc_index(n_start, P)))

    while open_set:
        _, ind = heapq.heappop(pq)
        if ind not in open_set:  # 중복 방지
            continue
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        del open_set[ind]

        if n_curr.x == round(gx / reso) and n_curr.y == round(gy / reso):
            break

        for dx, dy in P.motion:
            node = Node(n_curr.x + dx, n_curr.y + dy, n_curr.cost + u_cost([dx,dy]), ind)
            if not check_node(node, P, obsmap): continue
            n_ind = calc_index(node, P)
            if n_ind in closed_set: continue
            if n_ind not in open_set or open_set[n_ind].cost > node.cost:
                open_set[n_ind] = node
                heapq.heappush(pq, (node.cost, n_ind))

    pathx, pathy = extract_path(closed_set, n_start, n_goal, P)
    return pathx, pathy

# --- headless matplotlib 설정 & 결과 유틸 ---
import os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _ensure_results_dir(path="results"):
    os.makedirs(path, exist_ok=True)
    return path

def _stamp(prefix):
    return f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}.png"

def save_astar_figure(sx, sy, gx, gy, ox, oy, reso, rr, outdir="results"):
    _ensure_results_dir(outdir)
    ax, ay = astar_planning(sx, sy, gx, gy, ox, oy, reso, rr)
    plt.figure(figsize=(8,8))
    if ox and oy: plt.plot(ox, oy, 'sk', markersize=3, label='Obstacles')
    if ax and ay: plt.plot(ax, ay, '-r', linewidth=2, label='A*')
    plt.plot(sx, sy, 'og', label='Start'); plt.plot(gx, gy, 'ob', label='Goal')
    plt.axis("equal"); plt.grid(True); plt.legend(); plt.title("A*")
    fname = os.path.join(outdir, _stamp("astar"))
    plt.savefig(fname, dpi=180, bbox_inches="tight"); plt.close()

def save_dijkstra_figure(sx, sy, gx, gy, ox, oy, reso, rr, outdir="results"):
    _ensure_results_dir(outdir)
    dx, dy = dijkstra_planning(sx, sy, gx, gy, ox, oy, reso, rr)
    plt.figure(figsize=(8,8))
    if ox and oy: plt.plot(ox, oy, 'sk', markersize=3, label='Obstacles')
    if dx and dy: plt.plot(dx, dy, '-b', linewidth=2, label='Dijkstra')
    plt.plot(sx, sy, 'og', label='Start'); plt.plot(gx, gy, 'ob', label='Goal')
    plt.axis("equal"); plt.grid(True); plt.legend(); plt.title("Dijkstra")
    fname = os.path.join(outdir, _stamp("dijkstra"))
    plt.savefig(fname, dpi=180, bbox_inches="tight"); plt.close()

# def main():
#     sx = 10.0  # [m]
#     sy = 10.0  # [m]
#     gx = 50.0  # [m]
#     gy = 50.0  # [m]

#     robot_radius = 2.0
#     grid_resolution = 1.0
#     ox, oy = get_env()

#     pathx, pathy = astar_planning(sx, sy, gx, gy, ox, oy, grid_resolution, robot_radius)

#     plt.plot(ox, oy, 'sk')
#     plt.plot(pathx, pathy, '-r')
#     plt.plot(sx, sy, 'sg')
#     plt.plot(gx, gy, 'sb')
#     plt.axis("equal")
#     plt.show()

def main():
    sx, sy = 10.0, 10.0
    gx, gy = 50.0, 50.0
    robot_radius = 2.0
    grid_resolution = 1.0
    ox, oy = get_env()

    ax, ay = astar_planning(sx, sy, gx, gy, ox, oy, grid_resolution, robot_radius)
    dx, dy = dijkstra_planning(sx, sy, gx, gy, ox, oy, grid_resolution, robot_radius)

    plt.figure(figsize=(8,8))
    plt.plot(ox, oy, 'sk', label='Obstacles')
    plt.plot(ax, ay, '-r', linewidth=2, label='A*')
    plt.plot(dx, dy, '-b', linewidth=2, label='Dijkstra')
    plt.plot(sx, sy, 'og', label='Start'); plt.plot(gx, gy, 'ob', label='Goal')
    plt.axis("equal"); plt.grid(True); plt.legend(); plt.title("A* vs Dijkstra")
    plt.show()



if __name__ == '__main__':
    main()
