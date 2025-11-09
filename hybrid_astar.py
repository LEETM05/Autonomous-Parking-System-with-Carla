"""
Hybrid A*
@author: Huiming Zhou
"""

import os
import sys
import math
import heapq
from heapdict import heapdict
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd
import numba



import os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _ensure_results_dir(path="results"):
    os.makedirs(path, exist_ok=True)
    return path

def _stamp(prefix):
    return f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}.png"


sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import astar as astar
import draw as draw
import reeds_shepp as rs

try:
    import fastbridge as _fb
except ImportError:
    _fb = None

class C:  # Parameter config
    PI = math.pi

    XY_RESO = 0.2  # [m]
    YAW_RESO = np.deg2rad(1.0)  # [rad]
    MOVE_STEP = 0.2  # [m] path interporate resolution
    N_STEER = 20.0  # steer command number
    COLLISION_CHECK_STEP = 5  # skip number for collision check
    EXTEND_BOUND = 1  # collision check range extended

    GEAR_COST = 20.0  # switch back penalty cost
    BACKWARD_COST = 5.0  # backward penalty cost
    STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
    STEER_ANGLE_COST = 1.0  # steer angle penalty cost
    H_COST = 15.0  # Heuristic cost penalty cost
    
    HEURISTIC_WEIGHT = 2.0  # 휴리스틱 가중치 (1.0보다 큰 값으로 튜닝)

    # 1. 초기 탐색을 제한할 부채꼴의 각도 (단위: 도)
    INITIAL_SEARCH_ANGLE_DEG = 30.0

    # 2. 탐색이 막혔을 때, 몇 번의 반복마다 탐색 각도를 넓힐지 결정
    ANGLE_EXPANSION_ITER = 20000

    # 3. 각도를 넓힐 때 한 번에 몇 도씩 증가시킬지 결정
    ANGLE_EXPANSION_STEP_DEG = 15.0

    RF = 3.5  # [m] distance from rear to vehicle front end of vehicle

    RF = 3.5  # [m] distance from rear to vehicle front end of vehicle
    RB = 1.2  # [m] distance from rear to vehicle back end of vehicle
    W = 1.85  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 2.875  # [m] Wheel base
    TR = 0.5  # [m] Tyre radius
    TW = 1  # [m] Tyre width
    MAX_STEER = math.radians(30)  # [rad] maximum steering angle 

class Node:
    def __init__(self, xind, yind, yawind, direction, x, y,
                 yaw, directions, steer, cost, pind):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind


class Para:
    def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw,
                 xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree


class Path:
    def __init__(self, x, y, yaw, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        self.cost = cost


class QueuePrior:
    def __init__(self):
        self.queue = heapdict()

    def empty(self):
        return len(self.queue) == 0  # if Q is empty

    def put(self, item, priority):
        self.queue[item] = priority  # push 

    def get(self):
        return self.queue.popitem()[0]  # pop out element with smallest priority


# 각도 차이를 계산하는 헬퍼 함수를 추가
def angle_diff(angle1, angle2):
    """ 두 각도 사이의 최소 차이를 계산합니다 (-pi ~ pi) """
    return rs.pi_2_pi(angle1 - angle2)

def hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy, xyreso, yawreso):
    sxr, syr = round(sx / xyreso), round(sy / xyreso)
    gxr, gyr = round(gx / xyreso), round(gy / xyreso)
    syawr = round(rs.pi_2_pi(syaw) / yawreso)
    gyawr = round(rs.pi_2_pi(gyaw) / yawreso)

    nstart = Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [1], 0.0, 0.0, -1)
    ngoal = Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [1], 0.0, 0.0, -1)

    kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])
    P = calc_parameters(ox, oy, xyreso, yawreso, kdtree)

    hmap = astar.calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy, P.xyreso, 1.0)
    steer_set, direc_set = calc_motion_set()
    open_set, closed_set = {calc_index(nstart, P): nstart}, {}

    qp = QueuePrior()
    qp.put(calc_index(nstart, P), calc_hybrid_cost(nstart, hmap, P))
    
    
    # --- 탐색 각도 변수 및 상수 초기화 ---
    MAX_ITERATION = 200000  # 최대 반복 횟수도 조금 조정
    iter_count = 0
    
    # 논문에서 제안한 변수들을 초기화합니다.
    initial_search_angle = np.deg2rad(C.INITIAL_SEARCH_ANGLE_DEG)
    angle_expansion_step = np.deg2rad(C.ANGLE_EXPANSION_STEP_DEG)
    current_search_angle = initial_search_angle # 현재 적용되는 탐색 각도

    while True:
        iter_count += 1
        # --- 최대 반복 및 각도 확장 로직 ---
        if iter_count > MAX_ITERATION:
            print(f"Error: Maximum iteration {MAX_ITERATION} reached. Path not found.")
            return None, closed_set
        
        # 일정 반복마다 경로를 못찾으면 탐색 각도를 점차 넓힙니다.
        if iter_count % C.ANGLE_EXPANSION_ITER == 0:
            current_search_angle += angle_expansion_step
            print(f"Info: Path not found, expanding search angle to {np.rad2deg(current_search_angle):.1f} degrees.")
            # 최대 360도(전방향)까지만 넓어지도록 제한
            current_search_angle = min(current_search_angle, np.deg2rad(360.0))

        if not open_set:
            return None, closed_set

        ind = qp.get()
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        update, fpath = update_node_with_analystic_expantion(n_curr, ngoal, P)

        if update:
            fnode = fpath
            break

        # --- 유도 각도 계산 ---
        # 현재 노드에서 최종 목표(ngoal)까지의 방향을 계산
        guidance_angle = math.atan2(ngoal.y[-1] - n_curr.y[-1], ngoal.x[-1] - n_curr.x[-1])

        for i in range(len(steer_set)):
            # --- 각도 기반 필터링 ---
            # 1. 현재 조향각으로 움직였을 때의 예상 각도를 계산
            #    (calc_next_node 내부 로직을 간단히 모방)
            d = direc_set[i]
            u = steer_set[i]
            next_yaw = rs.pi_2_pi(n_curr.yaw[-1] + d * C.MOVE_STEP / C.WB * math.tan(u))
            
            # 2. 유도 각도와 예상 각도의 차이를 계산
            diff = abs(angle_diff(next_yaw, guidance_angle))

            # 3. 만약 차이가 허용된 탐색 각도의 절반보다 크면, 이 방향은 탐색하지 않고 건너뜀
            if diff > current_search_angle / 2.0:
                continue
            

            node = calc_next_node(n_curr, ind, u, d, P)

            if not node:
                continue

            node_ind = calc_index(node, P)

            if node_ind in closed_set:
                continue

            if node_ind not in open_set:
                open_set[node_ind] = node
                qp.put(node_ind, calc_hybrid_cost(node, hmap, P))
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node
                    qp.put(node_ind, calc_hybrid_cost(node, hmap, P))

    # return extract_path(closed_set, fnode, nstart)
    print("Final path found!")
    return extract_path(closed_set, fnode, nstart), closed_set # 성공 시에도 closed_set 반환


def extract_path(closed, ngoal, nstart):
    rx, ry, ryaw, direc = [], [], [], []
    cost = 0.0
    node = ngoal

    while True:
        rx += node.x[::-1]
        ry += node.y[::-1]
        ryaw += node.yaw[::-1]
        direc += node.directions[::-1]
        cost += node.cost

        if is_same_grid(node, nstart):
            break

        node = closed[node.pind]

    rx = rx[::-1]
    ry = ry[::-1]
    ryaw = ryaw[::-1]
    direc = direc[::-1]

    direc[0] = direc[1]
    path = Path(rx, ry, ryaw, direc, cost)

    return path

def calc_next_node(n_curr, c_id, u, d, P):
    step = C.XY_RESO * 2

    nlist = math.ceil(step / C.MOVE_STEP)
    xlist = [n_curr.x[-1] + d * C.MOVE_STEP * math.cos(n_curr.yaw[-1])]
    ylist = [n_curr.y[-1] + d * C.MOVE_STEP * math.sin(n_curr.yaw[-1])]
    yawlist = [rs.pi_2_pi(n_curr.yaw[-1] + d * C.MOVE_STEP / C.WB * math.tan(u))]

    for i in range(nlist - 1):
        xlist.append(xlist[i] + d * C.MOVE_STEP * math.cos(yawlist[i]))
        ylist.append(ylist[i] + d * C.MOVE_STEP * math.sin(yawlist[i]))
        yawlist.append(rs.pi_2_pi(yawlist[i] + d * C.MOVE_STEP / C.WB * math.tan(u)))

    xind = round(xlist[-1] / P.xyreso)
    yind = round(ylist[-1] / P.xyreso)
    yawind = round(yawlist[-1] / P.yawreso)

    if not is_index_ok(xind, yind, xlist, ylist, yawlist, P):
        return None

    cost = 0.0

    if d > 0:
        direction = 1
        cost += abs(step)
    else:
        direction = -1
        cost += abs(step) * C.BACKWARD_COST

    if direction != n_curr.direction:  # switch back penalty
        cost += C.GEAR_COST

    cost += C.STEER_ANGLE_COST * abs(u)  # steer angle penalyty
    cost += C.STEER_CHANGE_COST * abs(n_curr.steer - u)  # steer change penalty
    cost = n_curr.cost + cost

    directions = [direction for _ in range(len(xlist))]

    node = Node(xind, yind, yawind, direction, xlist, ylist,
                yawlist, directions, u, cost, c_id)

    return node


def is_index_ok(xind, yind, xlist, ylist, yawlist, P):
    if xind <= P.minx or \
            xind >= P.maxx or \
            yind <= P.miny or \
            yind >= P.maxy:
        return False

    ind = range(0, len(xlist), C.COLLISION_CHECK_STEP)

    nodex = [xlist[k] for k in ind]
    nodey = [ylist[k] for k in ind]
    nodeyaw = [yawlist[k] for k in ind]

    if is_collision(nodex, nodey, nodeyaw, P):
        return False

    return True


def update_node_with_analystic_expantion(n_curr, ngoal, P):
    path = analystic_expantion(n_curr, ngoal, P)  # rs path: n -> ngoal

    if not path:
        return False, None

    fx = path.x[1:-1]
    fy = path.y[1:-1]
    fyaw = path.yaw[1:-1]
    fd = path.directions[1:-1]

    fcost = n_curr.cost + calc_rs_path_cost(path)
    fpind = calc_index(n_curr, P)
    fsteer = 0.0

    fpath = Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction,
                 fx, fy, fyaw, fd, fsteer, fcost, fpind)

    return True, fpath


def analystic_expantion(node, ngoal, P):
    sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
    gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]

    maxc = math.tan(C.MAX_STEER) / C.WB
    paths = rs.calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=C.MOVE_STEP)

    if not paths:
        return None

    pq = QueuePrior()
    for path in paths:
        pq.put(path, calc_rs_path_cost(path))

    while not pq.empty():
        path = pq.get()
        ind = range(0, len(path.x), C.COLLISION_CHECK_STEP)

        pathx = [path.x[k] for k in ind]
        pathy = [path.y[k] for k in ind]
        pathyaw = [path.yaw[k] for k in ind]


        if not is_collision(pathx, pathy, pathyaw, P):
            return path

    return None

def is_collision(x, y, yaw, P):
    """
    x, y, yaw: 경로 샘플 배열(리스트/ndarray)
    P: 플래너 파라미터(ox, oy, kdtree, 차량 치수 C.RF/RB/W 등을 포함)
    fastbridge(path_collision_any)가 있으면 C++ 경로를 쓰고,
    없으면 기존 KDTree 기반 파이썬 경로로 폴백한다.
    """
    # --- 빠른 C++ 경로 (있으면 우선 사용) ---
    if _fb is not None:
        # 차량 직사각형 근사 파라미터
        rf = float(C.RF)
        rb = float(C.RB)
        half_w = float(C.W / 2.0)

        collided = _fb.path_collision_any(
            np.asarray(x, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            np.asarray(yaw, dtype=np.float32),
            np.asarray(P.ox, dtype=np.float32),
            np.asarray(P.oy, dtype=np.float32),
            rf, rb, half_w
        )
        return bool(collided)

    # --- 기존 파이썬/KDTree 경로 (폴백) ---
    # 차량을 회전 직사각으로 근사하여, KDTree로 근방 포인트만 후보로 가져와 충돌 판단
    for ix, iy, iyaw in zip(x, y, yaw):
        d = 0.1
        dl = (C.RF - C.RB) / 2.0
        r = (C.RF + C.RB) / 2.0 + d

        # 차량 중심을 앞/뒤 길이 평균만큼 앞쪽으로 이동한 점(차량 중심 근사)
        cx = ix + dl * math.cos(iyaw)
        cy = iy + dl * math.sin(iyaw)

        # KDTree에서 반경 r 내 후보 장애물 인덱스
        if hasattr(P, "kdtree") and P.kdtree is not None:
            ids = P.kdtree.query_ball_point([cx, cy], r)
        else:
            # KDTree가 없다면 전수 검사(느려질 수 있음)
            ids = range(len(P.ox))

        if not ids:
            continue

        for i in ids:
            # 차량 좌표계로 회전 변환 (yaw)
            xo = P.ox[i] - cx
            yo = P.oy[i] - cy
            dx =  xo * math.cos(iyaw) + yo * math.sin(iyaw)
            dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

            # 직사각 충돌 판정
            if (abs(dx) < r) and (abs(dy) < C.W / 2.0 + d):
                return True

    return False


def calc_rs_path_cost(rspath):
    cost = 0.0

    for lr in rspath.lengths:
        if lr >= 0:
            cost += 1
        else:
            cost += abs(lr) * C.BACKWARD_COST

    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            cost += C.GEAR_COST

    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += C.STEER_ANGLE_COST * abs(C.MAX_STEER)

    nctypes = len(rspath.ctypes)
    ulist = [0.0 for _ in range(nctypes)]

    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = -C.MAX_STEER
        elif rspath.ctypes[i] == "WB":
            ulist[i] = C.MAX_STEER

    for i in range(nctypes - 1):
        cost += C.STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    return cost


def calc_hybrid_cost(node, hmap, P):
    cost = node.cost + \
           C.H_COST * C.HEURISTIC_WEIGHT * hmap[node.xind - P.minx][node.yind - P.miny]

    return cost


def calc_motion_set():
    s = np.arange(C.MAX_STEER / C.N_STEER,
                  C.MAX_STEER, C.MAX_STEER / C.N_STEER)

    steer = list(s) + [0.0] + list(-s)
    direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
    steer = steer + steer

    return steer, direc


def is_same_grid(node1, node2):
    if node1.xind != node2.xind or \
            node1.yind != node2.yind or \
            node1.yawind != node2.yawind:
        return False

    return True


def calc_index(node, P):
    ind = (node.yawind - P.minyaw) * P.xw * P.yw + \
          (node.yind - P.miny) * P.xw + \
          (node.xind - P.minx)

    return ind


def calc_parameters(ox, oy, xyreso, yawreso, kdtree):
    minx = round(min(ox) / xyreso)
    miny = round(min(oy) / xyreso)
    maxx = round(max(ox) / xyreso)
    maxy = round(max(oy) / xyreso)

    xw, yw = maxx - minx, maxy - miny

    minyaw = round(-C.PI / yawreso) - 1
    maxyaw = round(C.PI / yawreso)
    yaww = maxyaw - minyaw

    return Para(minx, miny, minyaw, maxx, maxy, maxyaw,
                xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree)


def draw_car(x, y, yaw, steer, color='black'):
    car = np.array([[-C.RB, -C.RB, C.RF, C.RF, -C.RB],
                    [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])

    wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR],
                      [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[C.WB], [-C.WD / 2]])
    flWheel += np.array([[C.WB], [C.WD / 2]])
    rrWheel[1, :] -= C.WD / 2
    rlWheel[1, :] += C.WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)
    draw.Arrow(x, y, yaw, C.WB * 0.8, color)


def create_parking_lot_map(ox, oy, map_width, map_height, resolution=0.5):
    """
    주어진 크기에 맞춰 동적으로 주차장 맵을 생성합니다.
    """
    # 맵 외곽 경계
    for i in np.arange(0, map_width, resolution):
        ox.append(i); oy.append(0)
        ox.append(i); oy.append(map_height)
    for i in np.arange(0, map_height, resolution):
        ox.append(0); oy.append(i)
        ox.append(map_width); oy.append(i)

    # 주차 공간을 맵 중앙 상단에 배치
    center_x = map_width / 2.0
    park_y = map_height - 5.0
    
    obstacles_rect = [
        (center_x, park_y + 2.75, 8.0, 0.5),      # 주차장 뒷벽
        (center_x - 3.5, park_y, 2.0, 5.0),       # 왼쪽 주차 차량
        (center_x + 3.5, park_y, 2.0, 5.0),       # 오른쪽 주차 차량
    ]

    for r_x, r_y, w, h in obstacles_rect:
        for x in np.arange(r_x - w/2, r_x + w/2 + resolution, resolution):
            ox.extend([x, x]); oy.extend([r_y - h/2, r_y + h/2])
        for y in np.arange(r_y - h/2, r_y + h/2 + resolution, resolution):
            ox.extend([r_x - w/2, r_x + w/2]); oy.extend([y, y])
            
    return ox, oy, obstacles_rect

# ======================= visualization =======================

def visualize_initial_vs_final_save(sx, sy, syaw, gx, gy, gyaw, ox, oy, path, closed_set,
                                    xy_reso, robot_r, outdir="results"):
    """
    (1) Reeds–Shepp 초기 경로 (충돌 무시)
    (2) 2D A* 격자 초기 경로
    (3) Hybrid A* 최종 경로
    를 한 장에 겹쳐 저장한다.
    """
    _ensure_results_dir(outdir)
    from importlib import import_module
    _astar = import_module("astar")
    _rs = import_module("reeds_shepp")

    # RS 경로 (기구학적 초기 해)
    maxc = math.tan(C.MAX_STEER) / C.WB
    rs_only = _rs.calc_optimal_path(sx, sy, _rs.pi_2_pi(syaw), gx, gy, _rs.pi_2_pi(gyaw),
                                    maxc, step_size=C.MOVE_STEP)

    # A*(홀로노믹) 초기 경로
    ax, ay = _astar.astar_planning(sx, sy, gx, gy, ox, oy, xy_reso, robot_r)

    # 그림
    plt.figure(figsize=(10, 10))
    if ox and oy: plt.plot(ox, oy, ".k", label="Obstacles")
    if getattr(rs_only, "x", None):
        plt.plot(rs_only.x, rs_only.y, "--", linewidth=2, label="Initial (Reeds–Shepp)")
    if ax and ay:
        plt.plot(ax, ay, "b-.", linewidth=2, label="Initial (A* grid)")
    if path:
        plt.plot(path.x, path.y, "r-", linewidth=2.5, label="Final (Hybrid A*)")

    try:
        draw_car(sx, sy, syaw, 0.0, color='g')
        draw_car(gx, gy, gyaw, 0.0, color='r')
    except Exception:
        pass

    if closed_set:
        # 탐색 흔적(연한 회색)
        for node in closed_set.values():
            if node.pind != -1:
                plt.plot(node.x, node.y, "-", color='0.85', alpha=0.6)

    plt.title("Initial vs Final (Hybrid A*)")
    plt.axis("equal"); plt.grid(True); plt.legend()
    fname = os.path.join(_ensure_results_dir(outdir), _stamp("hybrid_astar"))
    plt.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close()

def save_hybrid_final_only(path, ox, oy, outdir="results"):
    """
    최종 Hybrid A* 경로만 단독 저장(이름: hybrid_astar_final_*.png)
    """
    _ensure_results_dir(outdir)
    plt.figure(figsize=(10,10))
    if ox and oy: plt.plot(ox, oy, ".k", label="Obstacles")
    if path:
        plt.plot(path.x, path.y, "r-", linewidth=2.5, label="Hybrid A* (final)")
    plt.axis("equal"); plt.grid(True); plt.legend()
    fname = os.path.join(outdir, _stamp("hybrid_astar_final"))
    plt.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close()


def save_hybrid_final_only(path, ox, oy, outdir="results"):
    """
    최종 Hybrid A* 경로만 단독 저장(이름: hybrid_astar_final_*.png)
    """
    _ensure_results_dir(outdir)
    plt.figure(figsize=(10,10))
    if ox and oy: plt.plot(ox, oy, ".k", label="Obstacles")
    if path:
        plt.plot(path.x, path.y, "r-", linewidth=2.5, label="Hybrid A* (final)")
    plt.axis("equal"); plt.grid(True); plt.legend()
    fname = os.path.join(outdir, _stamp("hybrid_astar_final"))
    plt.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close()



# --- 버전 1: 웨이포인트를 사용하는 main 함수 ---
def main():
    print("start! --- CARLA Data 2D Test ---")

    # 사용자가 CARLA 시뮬레이션에서 추출한 데이터
    start = [284.5531311035156, -233.98272705078125, 96.00579833984375]
    goal = [291.0, -229.5, 180.0]
    ox = [300.75955709815025, 299.76504585146904, 293.5990761220455, 296.20573117285966, 272.5363635018468, 271.54185225516557, 277.7287477940321, 273.94960505664346, 296.2475827917457, 278.36730616092683, 296.0905321612954, 279.7805477157235, 278.7860364690423, 281.98939826786517, 280.99488702118396, 293.54665453881023, 277.65540040135386, 275.28949921876193, 289.43248473107815, 293.63035777658223, 291.0446285352111, 277.7181778296828, 293.25347908735273, 292.2589678406715, 277.76002944856884, 296.8755711942911, 299.48222624510527, 298.48771499842405, 300.89546779990195, 299.90095655322074, 271.45793489813803, 270.4634236514568, 273.2689809516072, 272.274469704926, 274.4833202570677, 293.61978781223297, 278.70211911201477, 296.2264428630471, 280.7120674148202, 275.1428044334054, 281.9264067202806, 275.1846560522914, 275.2265076711774, 290.1651021808386, 277.6342604726553, 292.175050483644, 277.65518628209827, 277.6761120915413, 298.4037976413965, 297.4092863947153, 300.41374594420193, 299.4192346975207, 270.57840854376553, 277.75981532931326, 271.9916500985622, 296.0797480776906, 274.0015984013677, 273.0070871546865, 296.1215996965766, 275.2368635162711, 279.8325410604477, 278.83802981376647, 282.04139161258934, 281.0468803659081, 296.1843771249056, 293.61957369297744, 277.728319555521, 289.2855758264661, 291.69332862794397, 290.69881738126276, 296.09010392278435, 292.7087656840682, 296.11102973222734, 297.12646678835154, 296.1319555416703, 298.9375128418207, 297.9430015951395, 275.2681451708078, 299.9529498979449, 275.2890709802508, 270.9132214948535, 277.69682378172877, 272.92316979765894, 293.62992953807117, 274.53531360179187, 277.73867540061474, 279.1519169554114, 278.1574057087302, 293.4938047170639, 280.5651585102081, 277.7805270195007, 282.1773023143411, 293.5356563359499, 277.6444021984935, 277.66532800793647, 290.6149000242352, 289.620388777554, 293.6193595737219, 292.02814157903197, 275.12145038545134, 277.7281054362655, 277.74903124570847, 297.85908423811196, 296.0689639940858, 300.0679347902536, 299.0734235435724, 271.6249131351709, 270.6304018884897, 277.61290642470124, 273.03815468996766, 272.04364344328644, 275.04810299277307, 274.05359174609185, 296.17359304130076, 278.6701950997114, 277.6756838530302, 281.0779479011893, 280.0834366545081, 293.62971541881564, 277.7175354719162, 275.15273203998805, 296.0793198391795, 293.51451640725134, 289.93427591919897, 296.12117145806553, 291.94422422200444, 277.62326226979496, 293.1585635274649, 293.57729383558035, 297.9740691304207, 296.97955788373946, 300.18291968256233, 299.1884084358811, 277.7069655075669, 277.72789131700995, 270.5674103409052, 272.9751631423831, 271.9806518957019, 274.9851114451885, 273.9906001985073, 277.7906687453389, 278.60720355212686, 275.22586531341074, 280.4182496055961, 279.4237383589149, 282.4281979084015, 281.4336866617203, 293.5876496806741, 277.69639554321765, 296.23615635037424, 289.67238212227824, 275.1525179207325, 291.8812326744199, 290.8867214277387, 293.2944742292166, 275.1943695396185, 289.53625730127095, 297.9110775828361, 296.9165663361549, 291.54620560407636, 290.55169435739515, 289.55718311071394, 277.6230481505394, 299.32431913763287, 296.141883148253, 292.3627404108644, 291.36822916418316, 290.37371791750195, 289.37920667082074, 275.2571469679475, 293.5770797163248, 292.5825684696436, 291.5880572229624, 290.5935459762812, 289.59903472959996, 277.6648997694254, 293.5980055257678, 292.6034942790866, 291.6089830324054, 290.61447178572416, 289.61996053904295, 277.6858255788684, 293.6189313352108, 292.6244200885296, 291.62990884184836, 290.63539759516715, 289.64088634848594, 277.7067513883114, 273.1319996535778, 293.04315039664505, 292.04863914996383, 291.0541279032826, 290.0596166566014, 289.0651054099202, 293.26297845542433, 292.2684672087431, 291.2739559620619, 290.2794447153807, 289.2849334686995, 277.94750525653365, 293.0850020155311, 292.09049076884986, 291.09597952216865, 290.10146827548743, 289.1069570288062, 279.7585513100028, 277.7695288166404, 293.10592782497406, 292.11141657829285, 291.11690533161163, 290.1223940849304, 289.1278828382492, 281.37069511413574, 275.2047253847122, 292.9279513850808, 291.9334401383996, 290.9389288917184, 296.13131318390367, 292.35217044651506, 277.65432980507615, 289.6093905746937, 275.2884286224842, 291.42043662816286, 290.42592538148165, 293.23148268163203, 292.2369714349508, 277.7171072334051, 290.8655814990401, 289.8710702523589, 288.8765590056777, 296.45577028989794, 292.67662755250933, 291.6821163058281, 290.6876050591469, 289.6930938124657, 288.6985825657845, 271.39408687353136, 298.86352309137584, 296.07889160066844, 292.6975533619523, 291.7030421152711, 290.7085308685899, 289.7140196219087, 288.71950837522746, 273.20513292700053, 272.2106216803193, 292.91738142073154, 291.92287017405033, 290.9283589273691, 289.9338476806879, 288.9393364340067, 274.6183744817972, 293.336111728847, 292.3416004821658, 291.3470892354846, 290.3525779888034, 289.35806674212216, 296.1416690289974, 292.7603307902813, 291.76581954360006, 290.77130829691885, 289.77679705023763, 288.7822858035564, 278.63827108740804, 296.16259483844044, 292.7812565997243, 291.7867453530431, 290.7922341063619, 289.7977228596807, 288.80321161299946, 280.44931714087727, 293.5977914065123, 292.60328015983106, 291.60876891314985, 290.61425766646863, 289.6197464197874, 282.4592654436827, 293.61871721595526, 292.62420596927404, 291.62969472259283, 290.6351834759116, 289.6406722292304, 293.63964302539824, 292.4462295293808, 291.4517182826996, 290.45720703601836, 293.0638620868325, 292.0693508401513, 290.30015640556815, 296.08924744576217, 292.1112024590373, 296.11017325520515, 296.3300013139844, 297.9421451181173, 300.5488001689315, 299.5542889222503, 265.8977430462837, 266.13849691450594, 267.7506407186389, 270.1583935201168, 269.1638822734356, 271.37273282557726, 273.38268112838267, 275.3926294311881, 266.0442237123847, 276.6069687366486, 278.81581928879024, 265.88717308193446, 280.0301585942507, 282.4379113957286, 266.1279269501567, 283.85115295052526, 286.2589057520032, 285.264394505322, 287.6721473067999, 289.88099785894156, 288.88648661226034, 290.89643491506575, 292.90638321787117, 294.91633152067664, 296.92627982348205, 266.09643117636443, 297.94171687960625, 267.11186823248863, 300.5483719304204, 299.5538606837392, 302.1605157345533, 266.1592086046934, 303.7726595386863, 304.390292096138, 273.9582479476929, 266.9966692209244, 268.4099107757211, 304.432143715024, 268.82864108383654, 267.2583488985896, 281.6002366602421, 268.67159045338633, 304.494921143353, 268.51453982293606, 266.9233218282461, 267.54095438569783, 289.24222537279127, 288.24771412611005, 268.7552936911583, 304.578624381125, 268.5773172512651, 266.98609925657513, 267.8026340633631, 304.4215737506747, 295.8897028386593, 266.87090024501083, 268.8808485478163, 267.8863373011351, 270.294090102613, 304.32730054855347, 302.7360825538635, 271.9062339067459, 274.3139867082238, 273.3194754615426, 275.7272282630205, 304.3900779768825, 277.14046981781723, 279.3493203699589, 281.3592686727643, 304.45285540521144, 282.3747057288885, 284.58355628103016, 286.79240683317187, 285.79789558649065, 288.20564838796855, 304.3376563936472, 289.61888994276524, 292.02664274424313, 304.5784102618694, 293.4398842990398, 295.64873485118153, 304.4213596314192, 296.66417190730573, 298.8730224594474, 301.08187301158904, 304.48413705974815, 302.2962123170495, 303.90835612118246]
    oy = [-212.1673263192177, -212.27195537388326, -212.92065551280976, -212.84752223193647, -215.3376937329769, -215.44232278764247, -214.99252490997316, -215.39011531770228, -213.2453267544508, -215.32755199968815, -213.66405708789824, -215.37997358441353, -215.4846026390791, -215.34869192540646, -215.45332098007202, -214.33389715254307, -216.20686428844928, -216.65687628090382, -215.37004596590995, -215.12950619757174, -215.40154173970222, -216.8035710722208, -215.37026008069515, -215.4748891353607, -217.20137559473514, -215.39140000641345, -215.31826672554016, -215.42289578020572, -215.37068831026554, -215.4753173649311, -218.46770832836629, -218.57233738303185, -218.47827829122542, -218.582907345891, -218.55162568688394, -216.9405523598194, -218.5099881798029, -216.8674190789461, -218.49963233172895, -219.0855550378561, -218.57297972738743, -219.48335956037045, -219.8811640828848, -218.51063052415847, -219.8289566129446, -218.50027467608453, -220.02785887420177, -220.22676113545896, -218.44828132092954, -218.5529103755951, -218.43792547285557, -218.54255452752113, -221.57679711282253, -221.02237018048763, -221.6292186975479, -219.2960978358984, -221.61886284947394, -221.7234919041395, -219.69390235841274, -221.89111250638962, -221.6087211161852, -221.71335017085076, -221.57743945717812, -221.68206851184368, -220.29060914218425, -220.7615469455719, -222.63451408147813, -221.61971930861472, -221.56751183867453, -221.6721408933401, -221.30604625940322, -221.66178504526616, -221.5049485206604, -221.599221727252, -221.70385078191757, -221.60979169011117, -221.71442074477673, -224.09996319115163, -221.70406489670276, -224.2988654524088, -224.75923329293727, -224.2466579824686, -224.74887744486333, -222.77149536907672, -224.7803732186556, -224.64446250498295, -224.69688408970833, -224.8015131443739, -223.38912796378136, -224.7493056744337, -225.0422670274973, -224.780801448226, -223.7869324862957, -225.65989962220192, -225.85880188345908, -224.69752643406392, -224.80215548872948, -224.58254153132438, -224.7499480187893, -226.5286419481039, -226.45550866723062, -226.65441092848778, -224.73980628550052, -224.92813858389854, -224.70852462649344, -224.813153681159, -227.70091558992863, -227.8055446445942, -227.2720435231924, -227.753337174654, -227.85796622931957, -227.74298132658004, -227.8476103812456, -225.9226498901844, -227.76412125229837, -227.86875030696393, -227.71191378235818, -227.81654283702375, -226.5924899548292, -228.26655482947825, -228.7374926328659, -226.93808700740337, -227.409024810791, -227.78568940758706, -227.33589152991772, -227.7753335595131, -229.28199194669725, -227.84868095517157, -228.00573159456252, -227.7442660152912, -227.84889506995677, -227.71298435628415, -227.81761341094972, -230.07760099172592, -230.2765032529831, -231.02983244657517, -230.97762497663498, -231.08225403130055, -230.96726912856101, -231.07189818322658, -230.8732100367546, -230.98840905427932, -231.34414784014226, -230.99897901713848, -231.10360807180405, -230.9886231690645, -231.09325222373008, -230.01568001806737, -231.88864715397358, -230.3403512597084, -231.03090302050114, -232.55848721861838, -230.99962136149406, -231.10425041615963, -231.05204294621944, -232.95629174113273, -231.64853561520576, -230.96855381727218, -231.07318287193775, -231.63817976713182, -231.74280882179738, -231.84743787646295, -233.10298653244973, -231.02097540199756, -231.35578837692736, -231.75337878465652, -231.85800783932208, -231.96263689398765, -232.06726594865322, -233.55299852490424, -231.826726180315, -231.93135523498057, -232.03598428964614, -232.1406133443117, -232.24524239897727, -233.50079105496405, -232.0256284415722, -232.13025749623776, -232.23488655090333, -232.3395156055689, -232.44414466023446, -233.69969331622124, -232.22453070282936, -232.32915975749492, -232.4337888121605, -232.53841786682605, -232.64304692149162, -233.8985955774784, -234.37988922894002, -232.48621039688587, -232.59083945155143, -232.695468506217, -232.80009756088256, -232.90472661554813, -232.66418684720992, -232.76881590187548, -232.87344495654105, -232.97807401120662, -233.08270306587218, -234.27547428905964, -232.88401491940022, -232.98864397406578, -233.09327302873135, -233.1979020833969, -233.30253113806248, -234.28604425191878, -234.4953023612499, -233.08291718065738, -233.18754623532294, -233.2921752899885, -233.39680434465407, -233.50143339931964, -234.31754002571105, -234.96624016463755, -233.30274525284767, -233.40737430751324, -233.5120033621788, -233.16683453917503, -233.56442494690418, -235.31183721721172, -234.25519082248212, -235.76184920966625, -234.26576078534126, -234.37038984000682, -234.27633074820042, -234.380959802866, -235.90854400098323, -234.72634274065496, -234.83097179532052, -234.9356008499861, -234.33932229578494, -234.7369127035141, -234.84154175817966, -234.94617081284522, -235.0507998675108, -235.15542892217636, -236.9759744733572, -234.28711482584475, -234.58007617890834, -234.93581496477128, -235.04044401943685, -235.1450730741024, -235.24970212876798, -235.35433118343354, -236.98654443621635, -237.0911734908819, -235.11379141509533, -235.2184204697609, -235.32304952442647, -235.42767857909203, -235.5323076337576, -237.03896602094173, -235.27084205448628, -235.37547110915185, -235.4801001638174, -235.58472921848298, -235.68935827314854, -235.17678296267985, -235.5325217485428, -235.63715080320836, -235.74177985787392, -235.8464089125395, -235.95103796720505, -237.01825432479382, -235.37568522393704, -235.73142400979995, -235.83605306446552, -235.94068211913108, -236.04531117379665, -236.1499402284622, -237.02882428765298, -235.84662302732468, -235.95125208199025, -236.0558811366558, -236.16051019132138, -236.26513924598694, -237.018468439579, -236.04552528858184, -236.1501543432474, -236.25478339791297, -236.35941245257854, -236.4640415072441, -236.24442754983903, -236.3699824154377, -236.47461147010327, -236.57924052476884, -236.50610724389554, -236.6107362985611, -236.9979708582163, -236.59002460241317, -237.00854082107543, -236.78892686367035, -236.9669033139944, -236.99839908778668, -236.9252658069134, -237.02989486157895, -243.9895469725132, -244.36642568409442, -244.3979214578867, -244.3457139879465, -244.45034304261208, -244.419061383605, -244.40870553553106, -244.39834968745708, -245.3818628013134, -244.47169708311557, -244.4404154241085, -245.80059313476085, -244.513762819767, -244.46155534982682, -246.1774718463421, -244.5139769345522, -244.46176946461202, -244.56639851927758, -244.5141910493374, -244.48290939033032, -244.5875384449959, -244.57718259692192, -244.56682674884797, -244.556470900774, -244.54611505270003, -247.78961574733256, -244.64038825929165, -247.88388895392418, -244.56725497841836, -244.67188403308393, -244.5987507522106, -248.3863225311041, -244.63024652600288, -244.76637135446072, -247.96802042722703, -248.700423809886, -248.75284539461137, -245.16417587697507, -248.90989603400232, -249.2762047827244, -247.96844865679742, -249.32862636744977, -245.76088266074657, -249.74735670089723, -249.91476318836212, -250.05088801681995, -247.9688768863678, -248.07350594103337, -250.12423541247844, -246.55649170577527, -250.34406348466874, -250.51146997213363, -250.62666898965836, -246.97522203922273, -248.07393417060376, -251.32800482809543, -251.3176489800215, -251.42227803468705, -251.37007056474687, -247.9906591564417, -248.15806564390658, -251.4015663385391, -251.34935886859893, -251.4539879232645, -251.4017804533243, -248.5873659402132, -251.4542020380497, -251.42292037904264, -251.41256453096867, -249.1840727239847, -251.50683773756026, -251.4755560785532, -251.44427441954613, -251.5489034742117, -251.4966960042715, -250.00060757994652, -251.5491175889969, -251.4969101190567, -250.37748629152776, -251.5493317037821, -251.518050044775, -250.79621662497522, -251.6123232513666, -251.58104159235955, -251.54975993335248, -251.39292340874673, -251.62310732901096, -251.65460310280324]

    # Yaw 값을 라디안으로 변환 (플래너는 라디안 단위 사용)
    start[2] = np.deg2rad(start[2])
    goal[2] = np.deg2rad(goal[2])

    print("--- Planning from Start to Goal ---")
    t0 = time.time()
    
    # Hybrid A* 플래너를 호출하여 경로를 계산합니다.
    path, closed_set = hybrid_astar_planning(start[0], start[1], start[2], 
                                             goal[0], goal[1], goal[2],
                                             ox, oy, C.XY_RESO, C.YAW_RESO)
    
    t1 = time.time()
    print("Planning T: ", t1 - t0)

    # ======================= visualization =======================

    visualize_initial_vs_final(start[0], start[1], start[2], goal[0], goal[1], goal[2], ox, oy, path, closed_set)


    # --- 시각화 로직: 성공/실패 분리 ---
    if not path:
        print("Searching failed! Visualizing the setup...")
        plt.figure(figsize=(12, 12))
        plt.plot(ox, oy, ".k", label="Obstacles")
        # 접두사 제거 및 color 키워드 사용
        draw_car(start[0], start[1], start[2], 0.0, color='g')
        draw_car(goal[0], goal[1], goal[2], 0.0, color='r')
        
        if closed_set:
            for node in closed_set.values():
                if node.pind != -1:
                    plt.plot(node.x, node.y, "-", color='lightgray', alpha=0.5)

        plt.title("Hybrid A* (Failed Search)")
        plt.axis("equal"); plt.grid(True); plt.legend(); plt.show()
        return

    # --- 성공 시 최종 경로 생성 및 애니메이션 ---
    print("Found path! Visualizing final path animation...")

    for k in range(len(path.x)):
        plt.cla()
        plt.plot(ox, oy, ".k")
        plt.plot(path.x, path.y, linewidth=1.5, color='r', label="Hybrid A* Path")

        if k < len(path.x) - 2:
            dy = (path.yaw[k + 1] - path.yaw[k]) / C.MOVE_STEP
            steer = rs.pi_2_pi(math.atan(-C.WB * dy / path.direction[k]))
        else:
            steer = 0.0

        # draw. 접두사 제거 및 color 키워드 사용
        draw_car(goal[0], goal[1], goal[2], 0.0, color='dimgray')
        # color 키워드 사용
        draw_car(path.x[k], path.y[k], path.yaw[k], steer, color='black')
        
        plt.title("Hybrid A*")
        plt.axis("equal"); plt.grid(True); plt.legend(); plt.pause(0.001)
    
    plt.show()
    print("Done!")
    
if __name__ == '__main__':
    main()