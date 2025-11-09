import carla
import numpy as np
import math
import time
from queue import Empty
import subprocess
import json

# --- 분리된 모듈 임포트 ---
from carla_manager import CarlaManager
from pid_controller import PIDController, DebugPurePursuitController
from visualization import visualize_plan_debug
from carla_utils import get_goal_from_bev_pixel, get_parking_angle_from_bev, world_to_bev_pixels, generate_dynamic_parking_waypoint

# --- 플래너 및 관련 모듈 임포트 ---
import astar
import hybrid_astar
import reeds_shepp as rs
from pathlib import Path 

class Path:
    """Hybrid A* 경로 객체를 모방한 클래스 (경로 분할용)"""
    def __init__(self, x, y, yaw, direction, cost):
        self.x, self.y, self.yaw, self.direction, self.cost = x, y, yaw, direction, cost

def set_spectator_over_spawn(world,
                             x=285.0, y=-205.0, z=0.3,
                             yaw_deg=270.0, height=22.0,                
                             back=0.0, pitch=-90.0):               
    yaw_rad = math.radians(yaw_deg)
    dx = -back * math.cos(yaw_rad)
    dy = -back * math.sin(yaw_rad)
    loc = carla.Location(x=x + dx, y=y + dy, z=z + height)
    rot = carla.Rotation(pitch=pitch, yaw=yaw_deg, roll=0.0)
    world.get_spectator().set_transform(carla.Transform(loc, rot))

def main():
    manager = None
    try:
        class ParkingState:
            APPROACHING = 1
            PLANNING_PARK = 2
            PARKING = 3
        
        parking_state = ParkingState.APPROACHING
        manager = CarlaManager()
        player = manager.player
        speed_pid = PIDController(Kp=0.2, Ki=0.01, Kd=0.05)
        path_segments = []
        current_segment = None
        path_tracker_idx = 0
        frame_count = 0
        final_parking_transform = None
        approach_transform = None
        final_goal_location = None

        set_spectator_over_spawn(world=manager.world, x=285.0, y=-205.0, z=0.3, 
                                 yaw_deg=270.0, height=24.0, back=0.0, pitch=-90.0)
        st = time.time()
        while True:
            manager.world.tick()
            frame_count += 1
            if not player or not manager.process_bev_data_and_update_obstacles():
                continue
            
            ego_transform = player.get_transform()

            if frame_count == 1:
                try:
                    detection_image = manager.detection_queue.get(timeout=2.0)
                    detection_image.save_to_disk('./results/_temp_bev_image.png')
                except Empty:
                    print("FATAL: Could not get image from detection camera."); break
                
                command = ['python3.8', './detect_parking_spot/detect_parking_spot.py', '--image', './results/_temp_bev_image.png', '--model', './detect_parking_spot/best.pt', '--output_json', './results/result.json', '--output_vis_img', './results/_detected_parking_spot.png']
                subprocess.run(command)
                
                try:
                    with open('./results/result.json', 'r') as f:
                        detected_spots = json.load(f)
                        if not detected_spots:
                            print("FATAL: YOLO model did not detect any parking spots."); break
                        high_res_corners = detected_spots[0]['corners']
                except FileNotFoundError:
                    print("FATAL: result.json not found."); break
                
                scaled_down_corners = [[coord / 8.0 for coord in point] for point in high_res_corners]
                dynamic_yaw_deg = get_parking_angle_from_bev(scaled_down_corners, ego_transform)
                if dynamic_yaw_deg is None: break

                center_px = np.mean([c[0] for c in scaled_down_corners])
                center_py = np.mean([c[1] for c in scaled_down_corners])
                TARGET_BEV_PIXEL = (center_px, center_py)
                
                final_gx, final_gy, final_gyaw_rad = get_goal_from_bev_pixel(TARGET_BEV_PIXEL, ego_transform, dynamic_yaw_deg)
                final_parking_transform = carla.Transform(carla.Location(x=final_gx, y=final_gy), carla.Rotation(yaw=math.degrees(final_gyaw_rad)))
                approach_transform = generate_dynamic_parking_waypoint(final_parking_transform, ego_transform)

            # [ 1단계: 접근 ]
            if parking_state == ParkingState.APPROACHING:
                # [기존 is_approaching 로직을 이곳으로 이동]
                if approach_transform is None: continue

                dist_to_approach = ego_transform.location.distance(approach_transform.location)

                if dist_to_approach > 0.5:
                    target_speed_kmh = 2.0
                    target_yaw_rad = math.atan2(approach_transform.location.y - ego_transform.location.y,
                                               approach_transform.location.x - ego_transform.location.x)
                    ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
                    
                    # 목표와의 각도 차이를 계산하여 후진 여부를 결정합니다.
                    angle_diff = rs.pi_2_pi(target_yaw_rad - ego_yaw_rad)
                    
                    # 각도 차이가 90도(pi/2)를 초과하면 후진해야 함
                    should_reverse = abs(angle_diff) > (math.pi / 2)
                    
                    # 후진할 때는 목표 각도가 180도 반대가 되므로, 각도 오차를 다시 계산
                    if should_reverse:
                        steer_error = rs.pi_2_pi(target_yaw_rad - (ego_yaw_rad + math.pi))
                    else:
                        steer_error = angle_diff
                    
                    steer_output = np.clip(steer_error * 1.2, -1.0, 1.0)
                    current_velocity = player.get_velocity()
                    current_speed_mps = math.hypot(current_velocity.x, current_velocity.y)
                    speed_error = (target_speed_kmh / 3.6) - current_speed_mps
                    throttle_output = speed_pid.update(speed_error)
                    
                    control = carla.VehicleControl(
                        throttle=np.clip(throttle_output, 0, 0.5), 
                        steer=-steer_output,
                        brake=np.clip(-throttle_output, 0, 1.0),
                        reverse=should_reverse # ⭐️ 후진 플래그 적용
                    )
                    player.apply_control(control)
                    print(f"\rState: APPROACHING... Dist to Waypoint: {dist_to_approach:.2f}m", end="")
                else:
                    # [상태 변경] 접근 완료 -> 주차 계획으로 전환
                    print("\nApproach waypoint reached. Switching to main planning logic.")
                    player.apply_control(carla.VehicleControl(brake=1.0))
                    [manager.world.tick() for _ in range(30)]
                    parking_state = ParkingState.PLANNING_PARK
            
            # [ 2단계: 주차 경로 계획 ]
            elif parking_state == ParkingState.PLANNING_PARK:
                # [기존 needs_new_local_path 로직을 이곳으로 이동]
                rear_axle_location = carla.Location(
                    x=ego_transform.location.x - (2.875 / 2.0) * math.cos(math.radians(ego_transform.rotation.yaw)),
                    y=ego_transform.location.y - (2.875 / 2.0) * math.sin(math.radians(ego_transform.rotation.yaw)),
                )
                print("="*20)
                print(f"Attempting to plan path to perceived goal...")
                
                gx, gy = final_parking_transform.location.x, final_parking_transform.location.y
                gyaw = math.radians(final_parking_transform.rotation.yaw)
                final_goal_location = carla.Location(x=gx, y=gy)

                sx, sy = rear_axle_location.x, rear_axle_location.y
                syaw = math.radians(ego_transform.rotation.yaw)
                
                t0 = time.time()
                path, closed_set = hybrid_astar.hybrid_astar_planning(
                    sx, sy, syaw, gx, gy, gyaw,
                    manager.obstacle_ox, manager.obstacle_oy, 
                    hybrid_astar.C.XY_RESO, hybrid_astar.C.YAW_RESO
                )
                
                maxc = math.tan(hybrid_astar.C.MAX_STEER) / hybrid_astar.C.WB
                reeds_step = hybrid_astar.C.MOVE_STEP
                rs.save_reeds_shepp_figure(sx, sy, syaw, gx, gy, gyaw,
                                           max_curvature=maxc,
                                           step_size=reeds_step,
                                           outdir='results')
                
                astar.save_astar_figure(sx, sy, gx, gy, manager.obstacle_ox, manager.obstacle_oy,
                                        reso=1.0, rr=1.0, outdir='results')
                astar.save_dijkstra_figure(sx, sy, gx, gy, 
                                           manager.obstacle_ox, manager.obstacle_oy,
                                           reso=1.0, rr=1.0, outdir='results')

                hybrid_astar.visualize_initial_vs_final_save(
                    sx, sy, syaw, gx, gy, gyaw,
                    manager.obstacle_ox, manager.obstacle_oy,
                    path, closed_set,
                    xy_reso=hybrid_astar.C.XY_RESO, robot_r=1.0,
                    outdir='results'
                )
                hybrid_astar.save_hybrid_final_only(
                    path, manager.obstacle_ox, manager.obstacle_oy, outdir='results'
                )




                print(f"Path planning took: {time.time() - t0:.4f} seconds")

                goal_pixel_for_vis = world_to_bev_pixels(final_parking_transform.location, ego_transform)
                visualize_plan_debug(manager.bev_obstacle_map, goal_pixel_for_vis, path, closed_set, ego_transform, frame_count)
                
                if path:
                    print("SUCCESS: Path found!")
                    # [기존 경로 분할 로직 유지]
                    directions = path.direction
                    split_indices = [0] + [i for i in range(1, len(directions)) if directions[i] != directions[i-1]] + [len(directions)]
                    path_segments.clear()
                    for i in range(len(split_indices) - 1):
                        start_idx, end_idx = split_indices[i], split_indices[i+1]
                        segment = Path(path.x[start_idx:end_idx], path.y[start_idx:end_idx], 
                                       path.yaw[start_idx:end_idx], path.direction[start_idx:end_idx], 0.0)
                        path_segments.append(segment)
                    print(f"Path split into {len(path_segments)} segments.")
                    current_segment = path_segments.pop(0)
                    path_tracker_idx = 0
                    speed_pid.clear()
                    # [상태 변경] 주차 계획 완료 -> 주차 실행으로 전환
                    parking_state = ParkingState.PARKING
                else:
                    print("FAILURE: Path not found! Retrying...")
                    # 다시 계획 시도
                    parking_state = ParkingState.PLANNING_PARK 
                    player.apply_control(carla.VehicleControl(brake=1.0)); time.sleep(1)

            # [ 3단계: 주차 실행 ]
            elif parking_state == ParkingState.PARKING:
                # [기존 경로 추종 및 최종 정렬 로직을 모두 이곳으로 이동]
                ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
                wheelbase = 2.875
                rear_axle_location = carla.Location(
                    x=ego_transform.location.x - (wheelbase / 2.0) * math.cos(ego_yaw_rad),
                    y=ego_transform.location.y - (wheelbase / 2.0) * math.sin(ego_yaw_rad),
                )
                if not current_segment:
                    player.apply_control(carla.VehicleControl(brake=1.0))
                    continue

                is_final_maneuver = not path_segments
                if is_final_maneuver:
                    # [기존 최종 정렬 로직 유지]
                    dist_to_final = rear_axle_location.distance(final_goal_location)
                    goal_yaw_rad = math.radians(final_parking_transform.rotation.yaw)
                    yaw_error_rad = rs.pi_2_pi(ego_yaw_rad - goal_yaw_rad)
                    yaw_error_deg = math.degrees(yaw_error_rad)

                    if dist_to_final < 0.5 and abs(yaw_error_deg) < 0.5:
                        print(f"\nInitial parking position reached. Performing final adjustment...")
                        player.apply_control(carla.VehicleControl(brake=1.0, reverse=True))
                        for _ in range(30): manager.world.tick()
                        start_creep_location = player.get_location()
                        adjustment_distance = 1.0
                        while start_creep_location.distance(player.get_location()) < adjustment_distance:
                            control = carla.VehicleControl(throttle=0.25, steer=0.0, brake=0.0, reverse=True)
                            player.apply_control(control)
                            print(f"\rFinal Adjustment: Creeping backward...", end="")
                            manager.world.tick()
                        print(f"\nParking Complete! Final position locked.")
                        player.apply_control(carla.VehicleControl(brake=1.0, reverse=True))
                        time.sleep(5)
                        break

                # [기존 세그먼트 전환 로직 유지]
                last_point_in_segment = carla.Location(x=current_segment.x[-1], y=current_segment.y[-1])
                dist_to_last_point = rear_axle_location.distance(last_point_in_segment)
                if path_tracker_idx > len(current_segment.x) * 0.8 and dist_to_last_point < 0.5:
                    if path_segments:
                        print("\nSegment finished. Stopping and switching to next segment...")
                        player.apply_control(carla.VehicleControl(brake=1.0))
                        for _ in range(30): manager.world.tick()
                        current_segment = path_segments.pop(0)
                        path_tracker_idx = 0
                        speed_pid.clear()
                        continue
                    else:
                        print("\nAll path segments completed. Performing final alignment.")
                        pass
                
                # [기존 경로 추종 로직 유지]
                search_window_start = max(0, path_tracker_idx - 10)
                search_window_end = min(path_tracker_idx + 20, len(current_segment.x))
                min_dist, closest_idx_in_window = float('inf'), path_tracker_idx
                for i in range(search_window_start, search_window_end):
                    dist = math.hypot(rear_axle_location.x - current_segment.x[i], rear_axle_location.y - current_segment.y[i])
                    if dist < min_dist: min_dist, closest_idx_in_window = dist, i
                path_tracker_idx = closest_idx_in_window

                is_reverse = (current_segment.direction[path_tracker_idx] == -1)
                is_final_maneuver_flag = not path_segments
                if is_final_maneuver_flag and is_reverse:
                    lookahead_dist = 1.2
                    # target_speed_kmh = 0.5
                    target_speed_kmh = 2.0
                else:
                    lookahead_dist = 2.5
                    target_speed_kmh = 2.0
                            
                pp_controller = DebugPurePursuitController(wheelbase, 30.0) 
                world_path_for_pp = [{'location': carla.Location(x=px, y=py)} for px, py in zip(current_segment.x, current_segment.y)]
                steer_output = pp_controller.update(ego_transform, world_path_for_pp, is_reverse, lookahead_dist, path_tracker_idx)

                current_velocity = player.get_velocity()
                current_speed_mps = math.hypot(current_velocity.x, current_velocity.y)
                speed_error = (target_speed_kmh / 3.6) - current_speed_mps
                throttle_output = speed_pid.update(speed_error)
                
                throttle = np.clip(throttle_output, 0.0, 1.0)
                brake = np.clip(-throttle_output, 0.0, 1.0)
                final_steer = -steer_output
                
                control = carla.VehicleControl(throttle=throttle, steer=final_steer, brake=brake, reverse=is_reverse)
                
                print(f"\rFrame: {frame_count} | Segments Left: {len(path_segments)} | Reverse: {is_reverse} | Steer: {final_steer:.2f} | Speed Err: {speed_error:.2f}", end="")
                player.apply_control(control)

    finally:
        print("\nSimulation finished or error occurred.")
        if manager: manager.destroy()  
        print(f'Total Time : {time.time() - st:.4f}')

if __name__ == '__main__':
    main()