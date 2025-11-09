import numpy as np
import math
import cv2
import carla
import reeds_shepp as rs

# BEV 관련 전역 상수
BEV_RESOLUTION = 0.2
BEV_X_BOUND = [-10.0, 10.0, BEV_RESOLUTION]
BEV_Y_BOUND = [-10.0, 10.0, BEV_RESOLUTION]
BEV_SHAPE = (
    int((BEV_X_BOUND[1] - BEV_X_BOUND[0]) / BEV_X_BOUND[2]),
    int((BEV_Y_BOUND[1] - BEV_Y_BOUND[0]) / BEV_Y_BOUND[2])
)

def world_to_bev_pixels(world_point, ego_transform):
    world_p = np.array([world_point.x, world_point.y, world_point.z, 1.0])
    world_to_ego_matrix = np.array(ego_transform.get_inverse_matrix())
    ego_p = world_to_ego_matrix @ world_p
    
    local_x, local_y = ego_p[0], ego_p[1]
    
    pixel_y = int(BEV_SHAPE[0] / 2 - local_x / BEV_X_BOUND[2])
    pixel_x = int(BEV_SHAPE[1] / 2 + local_y / BEV_Y_BOUND[2])
    
    return pixel_x, pixel_y

def convert_world_to_ego(world_x, world_y, world_yaw, ego_transform):
    world_p = np.array([world_x, world_y])
    ego_location = ego_transform.location
    ego_yaw_rad = np.deg2rad(ego_transform.rotation.yaw)
    
    R = np.array([
        [math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)],
        [-math.sin(ego_yaw_rad), math.cos(ego_yaw_rad)]
    ])
    
    local_p = R @ (world_p - np.array([ego_location.x, ego_location.y]))
    local_x, local_y = local_p[0], local_p[1]
    
    world_yaw_rad = np.deg2rad(world_yaw)
    local_yaw = rs.pi_2_pi(world_yaw_rad - ego_yaw_rad)
    
    return local_x, local_y, local_yaw

def get_goal_from_bev_pixel(target_pixel_xy, ego_transform, target_world_yaw_deg):
    px, py = target_pixel_xy
    local_y = (px - BEV_SHAPE[1] / 2) * BEV_Y_BOUND[2]
    local_x = (BEV_SHAPE[0] / 2 - py) * BEV_X_BOUND[2]

    local_point = np.array([local_x, local_y, 0.0, 1.0])
    ego_to_world_matrix = np.array(ego_transform.get_matrix())
    world_point = ego_to_world_matrix @ local_point
    
    gx = world_point[0]
    gy = world_point[1]
    gyaw = np.deg2rad(target_world_yaw_deg)

    print(f"Target BEV Pixel ({px}, {py}) -> World Goal ({gx:.2f}, {gy:.2f})")
    return gx, gy, gyaw

def get_parking_angle_from_bev(four_corner_points, ego_transform):
    """주차 공간의 네 꼭짓점(BEV 이미지)에서 월드 좌표계의 최적 주차 yaw를 계산합니다."""
    # ... (run_carla_parking.py에 있던 원본 함수 내용 전체를 여기에 붙여넣습니다) ...
    if len(four_corner_points) != 4:
        print("Error: Exactly 4 corner points required.")
        return None
    points = np.array(four_corner_points, dtype=np.float32)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean=None)
    if eigenvectors is None or len(eigenvectors) == 0:
        print("Error: PCA computation failed.")
        return None
    vx, vy = eigenvectors[0]
    axis_angle_rad = math.atan2(vy, vx)
    axis_angle_deg = math.degrees(axis_angle_rad)
    slot_yaw_1 = (axis_angle_deg) % 360
    slot_yaw_2 = (axis_angle_deg + 180) % 360
    if slot_yaw_1 > 180: slot_yaw_1 -= 360
    if slot_yaw_2 > 180: slot_yaw_2 -= 360
    center_px = np.mean(points[:, 0])
    center_py = np.mean(points[:, 1])
    local_y = (center_px - BEV_SHAPE[1] / 2) * BEV_Y_BOUND[2]
    local_x = (BEV_SHAPE[0] / 2 - center_py) * BEV_X_BOUND[2]
    local_point = np.array([local_x, local_y, 0.0, 1.0])
    ego_to_world_matrix = np.array(ego_transform.get_matrix())
    world_point = ego_to_world_matrix @ local_point
    spot_center_world = carla.Location(x=world_point[0], y=world_point[1])
    ego_location = ego_transform.location
    approach_vector = spot_center_world - ego_location
    approach_angle_deg = math.degrees(math.atan2(approach_vector.y, approach_vector.x)) % 360
    if approach_angle_deg > 180: approach_angle_deg -= 360
    def angle_between(a, b): return min((a - b) % 360, (b - a) % 360)
    diff1 = angle_between(approach_angle_deg, slot_yaw_1)
    diff2 = angle_between(approach_angle_deg, slot_yaw_2)
    final_yaw_deg = slot_yaw_1 if diff1 > diff2 else slot_yaw_2
    if final_yaw_deg > 180: final_yaw_deg -= 360
    elif final_yaw_deg < -180: final_yaw_deg += 360
    snapped_yaw = round(final_yaw_deg / 90.0) * 90.0
    if snapped_yaw > 180: snapped_yaw -= 360
    elif snapped_yaw < -180: snapped_yaw += 360
    print(f"Final Yaw: {final_yaw_deg:.1f}° -> Snapped: {snapped_yaw:.1f}°")
    return snapped_yaw

def generate_dynamic_parking_waypoint(final_transform, start_transform):
    final_loc = final_transform.location
    waypoint_loc = carla.Location(x=start_transform.location.x, y=final_loc.y, z=final_loc.z)
    waypoint_yaw_deg = 90.0 if start_transform.location.y < final_loc.y else 270.0
    waypoint_transform = carla.Transform(waypoint_loc, carla.Rotation(yaw=waypoint_yaw_deg))
    print(f"Ideal Dynamic Waypoint Generated at: {waypoint_loc}, Yaw: {waypoint_yaw_deg}")
    return waypoint_transform
