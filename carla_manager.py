# carla_manager.py

import carla
from queue import Queue, Empty
import numpy as np
import math
import cv2
import random
from carla_utils import BEV_SHAPE, BEV_X_BOUND, BEV_Y_BOUND, world_to_bev_pixels

class CarlaManager:
    def __init__(self):
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town04_Opt')
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds, settings.synchronous_mode = 1.0 / 30.0, True
        self.world.apply_settings(settings)

        self.player, self.bev_camera, self.actor_list = None, None, []
        self.bev_queue = Queue()
        self.detection_camera = None
        self.detection_queue = Queue()
        self.bev_obstacle_map = np.zeros(BEV_SHAPE, dtype=np.uint8)
        self.obstacle_ox = []
        self.obstacle_oy = []

        self.spawn_actors()
        self.spawn_static_vehicles()

    def spawn_actors(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        start_transform = carla.Transform(carla.Location(x=285.0, y=-205.0, z=0.3), carla.Rotation(yaw=270.0))
        self.player = self.world.try_spawn_actor(vehicle_bp, start_transform)
        if self.player is None: exit("Error: Could not spawn player vehicle.")
        
        physics_control = self.player.get_physics_control()
        physics_control.wheels[0].max_steer_angle = 30.0
        physics_control.wheels[1].max_steer_angle = 30.0
        self.player.apply_physics_control(physics_control)
        self.actor_list.append(self.player)

        cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        cam_bp.set_attribute('image_size_x', str(BEV_SHAPE[1]))
        cam_bp.set_attribute('image_size_y', str(BEV_SHAPE[0]))
        cam_bp.set_attribute('fov', str(2 * math.degrees(math.atan2(BEV_X_BOUND[1], 40))))
        cam_transform = carla.Transform(carla.Location(z=40), carla.Rotation(pitch=-90))
        self.bev_camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.player)
        self.bev_camera.listen(self.bev_queue.put)
        self.actor_list.append(self.bev_camera)
        
        high_res_cam_bp = blueprint_library.find('sensor.camera.rgb')
        high_res_cam_bp.set_attribute('image_size_x', '800')
        high_res_cam_bp.set_attribute('image_size_y', '800')
        cam_height = 40
        fov = str(2 * math.degrees(math.atan2(BEV_X_BOUND[1], cam_height)))
        high_res_cam_bp.set_attribute('fov', fov)
        cam_transform = carla.Transform(carla.Location(z=cam_height), carla.Rotation(pitch=-90))
        self.detection_camera = self.world.spawn_actor(high_res_cam_bp, cam_transform, attach_to=self.player)
        self.detection_camera.listen(self.detection_queue.put)
        self.actor_list.append(self.detection_camera)

    def process_bev_data_and_update_obstacles(self):
        try:
            bev_image = self.bev_queue.get(block=True, timeout=1.0)
        except Empty:
            print("Warning: BEV sensor timeout."); return False

        data = np.frombuffer(bev_image.raw_data, dtype=np.uint8).reshape((bev_image.height, bev_image.width, 4))
        semantic_image = data[:, :, 2]
        
        cost_map = np.ones_like(semantic_image, dtype=np.uint8) * 5
        cost_map[semantic_image == 7] = 0
        cost_map[semantic_image == 14] = 0
        cost_map[semantic_image == 8] = 155
        cost_map[semantic_image == 6] = 255
        cost_map[semantic_image == 10] = 255
        cost_map[semantic_image == 4] = 255
        self.bev_obstacle_map = cost_map

        ego_bb = self.player.bounding_box
        ego_transform = self.player.get_transform()
        vertices = [v for v in ego_bb.get_world_vertices(ego_transform)]
        bottom_vertices = sorted(vertices, key=lambda v: v.z)[:4]
        ego_footprint_pixels = [world_to_bev_pixels(v, ego_transform) for v in bottom_vertices]
        
        pixel_array = np.array(ego_footprint_pixels)
        centroid = np.mean(pixel_array, axis=0)
        sorted_pixels = sorted(ego_footprint_pixels, key=lambda p: math.atan2(p[1] - centroid[1], p[0] - centroid[0]))
        cv2.fillPoly(self.bev_obstacle_map, [np.array(sorted_pixels, dtype=np.int32)], 0)

        obstacle_pixels = np.argwhere(self.bev_obstacle_map >= 200)
        downsampled_pixels = obstacle_pixels[::5]
        new_ox, new_oy = self.bev_pixels_to_world(downsampled_pixels)
        self.obstacle_ox = new_ox
        self.obstacle_oy = new_oy
        return True
    
    def bev_pixels_to_world(self, pixels):
        ego_transform = self.player.get_transform()
        ego_to_world_matrix = np.array(ego_transform.get_matrix())
        ox, oy = [], []
        for p in pixels:
            local_y = (p[1] - BEV_SHAPE[1] / 2) * BEV_Y_BOUND[2]
            local_x = (BEV_SHAPE[0] / 2 - p[0]) * BEV_X_BOUND[2]
            local_p = np.array([local_x, local_y, 0.0, 1.0])
            world_p = ego_to_world_matrix @ local_p
            ox.append(world_p[0])
            oy.append(world_p[1])
        return ox, oy
 
    def spawn_static_vehicles(self):
        """
        미리 정의된 모든 주차 공간에 40% 확률로 장애물 차량을
        무작위로 정적 스폰합니다.
        """
        # --- 1. Town04의 모든 주차 공간 위치를 Transform 리스트로 정의 ---
        ALL_PARKING_SPOTS = [
                # row 1
            carla.Transform(carla.Location(x=298.5, y=-235.73, z=0.3), carla.Rotation(yaw=0.0)),  # 1-1
            carla.Transform(carla.Location(x=298.5, y=-232.73, z=0.3), carla.Rotation(yaw=0.0)),  # 1-2
            carla.Transform(carla.Location(x=298.5, y=-229.53, z=0.3), carla.Rotation(yaw=0.0)),  # 1-3
            carla.Transform(carla.Location(x=298.5, y=-226.43, z=0.3), carla.Rotation(yaw=0.0)),  # 1-4
            carla.Transform(carla.Location(x=298.5, y=-223.43, z=0.3), carla.Rotation(yaw=0.0)),  # 1-5
            carla.Transform(carla.Location(x=298.5, y=-220.23, z=0.3), carla.Rotation(yaw=0.0)),  # 1-6
            carla.Transform(carla.Location(x=298.5, y=-217.23, z=0.3), carla.Rotation(yaw=0.0)),  # 1-7
            carla.Transform(carla.Location(x=298.5, y=-214.03, z=0.3), carla.Rotation(yaw=0.0)),  # 1-8
            carla.Transform(carla.Location(x=298.5, y=-210.73, z=0.3), carla.Rotation(yaw=0.0)),  # 1-9
            carla.Transform(carla.Location(x=298.5, y=-207.30, z=0.3), carla.Rotation(yaw=0.0)), # 1-10
            carla.Transform(carla.Location(x=298.5, y=-204.23, z=0.3), carla.Rotation(yaw=0.0)), # 1-11
            carla.Transform(carla.Location(x=298.5, y=-201.03, z=0.3), carla.Rotation(yaw=0.0)), # 1-12
            carla.Transform(carla.Location(x=298.5, y=-198.03, z=0.3), carla.Rotation(yaw=0.0)), # 1-13
            carla.Transform(carla.Location(x=298.5, y=-194.90, z=0.3), carla.Rotation(yaw=0.0)), # 1-14
            carla.Transform(carla.Location(x=298.5, y=-191.53, z=0.3), carla.Rotation(yaw=0.0)), # 1-15
            carla.Transform(carla.Location(x=298.5, y=-188.20, z=0.3), carla.Rotation(yaw=0.0)), # 1-16

            # row 2
            carla.Transform(carla.Location(x=290.9, y=-235.73, z=0.3), carla.Rotation(yaw=180.0)), # 2-1
            carla.Transform(carla.Location(x=290.9, y=-232.73, z=0.3), carla.Rotation(yaw=180.0)), # 2-2
            carla.Transform(carla.Location(x=290.9, y=-229.53, z=0.3), carla.Rotation(yaw=180.0)), # 2-3
            carla.Transform(carla.Location(x=290.9, y=-226.43, z=0.3), carla.Rotation(yaw=180.0)), # 2-4
            carla.Transform(carla.Location(x=290.9, y=-223.43, z=0.3), carla.Rotation(yaw=180.0)), # 2-5
            carla.Transform(carla.Location(x=290.9, y=-220.23, z=0.3), carla.Rotation(yaw=180.0)), # 2-6
            carla.Transform(carla.Location(x=290.9, y=-217.23, z=0.3), carla.Rotation(yaw=180.0)), # 2-7
            carla.Transform(carla.Location(x=290.9, y=-214.03, z=0.3), carla.Rotation(yaw=180.0)), # 2-8
            carla.Transform(carla.Location(x=290.9, y=-210.73, z=0.3), carla.Rotation(yaw=180.0)), # 2-9
            carla.Transform(carla.Location(x=290.9, y=-207.30, z=0.3), carla.Rotation(yaw=180.0)), # 2-10
            carla.Transform(carla.Location(x=290.9, y=-204.23, z=0.3), carla.Rotation(yaw=180.0)), # 2-11
            carla.Transform(carla.Location(x=290.9, y=-201.03, z=0.3), carla.Rotation(yaw=180.0)), # 2-12
            carla.Transform(carla.Location(x=290.9, y=-198.03, z=0.3), carla.Rotation(yaw=180.0)), # 2-13
            carla.Transform(carla.Location(x=290.9, y=-194.90, z=0.3), carla.Rotation(yaw=180.0)), # 2-14
            carla.Transform(carla.Location(x=290.9, y=-191.53, z=0.3), carla.Rotation(yaw=180.0)), # 2-15
            carla.Transform(carla.Location(x=290.9, y=-188.20, z=0.3), carla.Rotation(yaw=180.0)), # 2-16

            # row 3
            carla.Transform(carla.Location(x=280.0, y=-235.73, z=0.3), carla.Rotation(yaw=0.0)),  # 3-1
            carla.Transform(carla.Location(x=280.0, y=-232.73, z=0.3), carla.Rotation(yaw=0.0)),   # 3-2
            carla.Transform(carla.Location(x=280.0, y=-229.53, z=0.3), carla.Rotation(yaw=0.0)), # 3-3
            carla.Transform(carla.Location(x=280.0, y=-226.43, z=0.3), carla.Rotation(yaw=0.0)), # 3-4
            carla.Transform(carla.Location(x=280.0, y=-223.43, z=0.3), carla.Rotation(yaw=0.0)), # 3-5
            carla.Transform(carla.Location(x=280.0, y=-220.23, z=0.3), carla.Rotation(yaw=0.0)), # 3-6
            carla.Transform(carla.Location(x=280.0, y=-217.23, z=0.3), carla.Rotation(yaw=0.0)), # 3-7
            carla.Transform(carla.Location(x=280.0, y=-214.03, z=0.3), carla.Rotation(yaw=0.0)), # 3-8
            carla.Transform(carla.Location(x=280.0, y=-210.73, z=0.3), carla.Rotation(yaw=0.0)), # 3-9
            carla.Transform(carla.Location(x=280.0, y=-207.30, z=0.3), carla.Rotation(yaw=0.0)), # 3-10
            carla.Transform(carla.Location(x=280.0, y=-204.23, z=0.3), carla.Rotation(yaw=0.0)), # 3-11
            carla.Transform(carla.Location(x=280.0, y=-201.03, z=0.3), carla.Rotation(yaw=0.0)), # 3-12
            carla.Transform(carla.Location(x=280.0, y=-198.03, z=0.3), carla.Rotation(yaw=0.0)), # 3-13
            carla.Transform(carla.Location(x=280.0, y=-194.90, z=0.3), carla.Rotation(yaw=0.0)), # 3-14
            carla.Transform(carla.Location(x=280.0, y=-191.53, z=0.3), carla.Rotation(yaw=0.0)), # 3-15
            carla.Transform(carla.Location(x=280.0, y=-188.20, z=0.3), carla.Rotation(yaw=0.0)), # 3-16

            # row 4
            carla.Transform(carla.Location(x=272.5, y=-235.73, z=0.3), carla.Rotation(yaw=180.0)),  # 4-1
            carla.Transform(carla.Location(x=272.5, y=-232.73, z=0.3), carla.Rotation(yaw=180.0)),  # 4-2
            carla.Transform(carla.Location(x=272.5, y=-229.53, z=0.3), carla.Rotation(yaw=180.0)),  # 4-3
            carla.Transform(carla.Location(x=272.5, y=-226.43, z=0.3), carla.Rotation(yaw=180.0)),  # 4-4
            carla.Transform(carla.Location(x=272.5, y=-223.43, z=0.3), carla.Rotation(yaw=180.0)), # 4-5
            carla.Transform(carla.Location(x=272.5, y=-220.23, z=0.3), carla.Rotation(yaw=180.0)), # 4-6
            carla.Transform(carla.Location(x=272.5, y=-217.23, z=0.3), carla.Rotation(yaw=180.0)), # 4-7
            carla.Transform(carla.Location(x=272.5, y=-214.03, z=0.3), carla.Rotation(yaw=180.0)), # 4-8
            carla.Transform(carla.Location(x=272.5, y=-210.73, z=0.3), carla.Rotation(yaw=180.0)), # 4-9
            carla.Transform(carla.Location(x=272.5, y=-207.30, z=0.3), carla.Rotation(yaw=180.0)), # 4-10
            carla.Transform(carla.Location(x=272.5, y=-204.23, z=0.3), carla.Rotation(yaw=180.0)), # 4-11
            carla.Transform(carla.Location(x=272.5, y=-201.03, z=0.3), carla.Rotation(yaw=180.0)), # 4-12
            carla.Transform(carla.Location(x=272.5, y=-198.03, z=0.3), carla.Rotation(yaw=180.0)), # 4-13
            carla.Transform(carla.Location(x=272.5, y=-194.90, z=0.3), carla.Rotation(yaw=180.0)), # 4-14
            carla.Transform(carla.Location(x=272.5, y=-191.53, z=0.3), carla.Rotation(yaw=180.0)), # 4-15
            carla.Transform(carla.Location(x=272.5, y=-188.20, z=0.3), carla.Rotation(yaw=180.0)), # 4-16
        ]

        # --- 2. 차량 필터링 ---
        blacklist = [
            'vehicle.carlamotors.carlacola', 'vehicle.tesla.cybertruck', 'vehicle.carlamotors.firetruck',
            'vehicle.mercedes.sprinter', 'vehicle.ford.ambulance', 'vehicle.nissan.patrol_2021',
            'vehicle.chevrolet.impala', 'vehicle.tesla.model3'
        ]
        all_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [bp for bp in all_blueprints
                      if bp.id not in blacklist and int(bp.get_attribute('number_of_wheels')) == 4]
        
        # --- 3. 90% 확률로 무작위 스폰 ---
        for spot_transform in ALL_PARKING_SPOTS:
            # 확률을 90%로 조정합니다.
            if random.random() < 0.9:
                try:
                    vehicle = self.world.try_spawn_actor(random.choice(blueprints), spot_transform)
                    if vehicle:
                        vehicle.set_simulate_physics(False)
                        self.actor_list.append(vehicle)
                except Exception:
                    pass
        print(f"Spawned obstacle vehicles.")

    def destroy(self):
        print("Destroying actors...")
        for actor in self.actor_list:
            if actor and actor.is_alive:
                actor.destroy()
        print("All actors destroyed.")



# # carla_manager.py

# import carla
# from queue import Queue, Empty
# import numpy as np
# import math
# import cv2
# import random
# from carla_utils import BEV_SHAPE, BEV_X_BOUND, BEV_Y_BOUND, world_to_bev_pixels

# class CarlaManager:
#     def __init__(self):
#         self.client = carla.Client('127.0.0.1', 2000)
#         self.client.set_timeout(10.0)
#         self.world = self.client.load_world('Town04_Opt')
#         self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
#         settings = self.world.get_settings()
#         settings.fixed_delta_seconds, settings.synchronous_mode = 1.0 / 30.0, True
#         self.world.apply_settings(settings)

#         self.player, self.bev_camera, self.actor_list = None, None, []
#         self.bev_queue = Queue()
#         self.detection_camera = None
#         self.detection_queue = Queue()
#         self.bev_obstacle_map = np.zeros(BEV_SHAPE, dtype=np.uint8)
#         self.obstacle_ox = []
#         self.obstacle_oy = []

#         self._sem_saved_once = False                 
#         self._sem_save_path_png = "semantic_once.png" 
#         self._sem_save_path_npy = "semantic_once_labels.npy" 

#         self.spawn_actors()
#         self.spawn_static_vehicles()

#     def spawn_actors(self):
#         blueprint_library = self.world.get_blueprint_library()
#         vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
#         start_transform = carla.Transform(carla.Location(x=285.0, y=-205.0, z=0.3), carla.Rotation(yaw=270.0))
#         self.player = self.world.try_spawn_actor(vehicle_bp, start_transform)
#         if self.player is None: exit("Error: Could not spawn player vehicle.")
        
#         physics_control = self.player.get_physics_control()
#         physics_control.wheels[0].max_steer_angle = 30.0
#         physics_control.wheels[1].max_steer_angle = 30.0
#         self.player.apply_physics_control(physics_control)
#         self.actor_list.append(self.player)

#         cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
#         # cam_bp.set_attribute('image_size_x', str(BEV_SHAPE[1]))
#         # cam_bp.set_attribute('image_size_y', str(BEV_SHAPE[0]))
#         cam_bp.set_attribute('image_size_x', '1024')
#         cam_bp.set_attribute('image_size_y', '1024')
#         cam_bp.set_attribute('fov', str(2 * math.degrees(math.atan2(BEV_X_BOUND[1], 40))))
#         cam_transform = carla.Transform(carla.Location(z=40), carla.Rotation(pitch=-90))
#         self.bev_camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.player)
#         self.bev_camera.listen(self.bev_queue.put)
#         self.actor_list.append(self.bev_camera)
        
#         high_res_cam_bp = blueprint_library.find('sensor.camera.rgb')
#         high_res_cam_bp.set_attribute('image_size_x', '800')
#         high_res_cam_bp.set_attribute('image_size_y', '800')
#         cam_height = 40
#         fov = str(2 * math.degrees(math.atan2(BEV_X_BOUND[1], cam_height)))
#         high_res_cam_bp.set_attribute('fov', fov)
#         cam_transform = carla.Transform(carla.Location(z=cam_height), carla.Rotation(pitch=-90))
#         self.detection_camera = self.world.spawn_actor(high_res_cam_bp, cam_transform, attach_to=self.player)
#         self.detection_camera.listen(self.detection_queue.put)
#         self.actor_list.append(self.detection_camera)

#     def process_bev_data_and_update_obstacles(self):
#         try:
#             bev_image = self.bev_queue.get(block=True, timeout=1.0)
#         except Empty:
#             print("Warning: BEV sensor timeout."); return False

#         data = np.frombuffer(bev_image.raw_data, dtype=np.uint8).reshape((bev_image.height, bev_image.width, 4))
#         semantic_image = data[:, :, 2]
        

#         if not self._sem_saved_once:
#             try:
#                 # 보기용(컬러 팔레트) PNG: CARLA가 제공하는 CityScapes 팔레트로 렌더
#                 bev_image.save_to_disk(self._sem_save_path_png, carla.ColorConverter.CityScapesPalette)
#             except Exception as e:
#                 print(f"[WARN] semantic PNG save failed: {e}")

#             # 라벨 ID 원본도 함께 저장(후처리/학습용)
#             # 아래 두 줄은 semantic_image 생성 이후에 넣어도 됩니다.
#             data = np.frombuffer(bev_image.raw_data, dtype=np.uint8).reshape((bev_image.height, bev_image.width, 4))
#             semantic_image_once = data[:, :, 2]  # 라벨 ID (8비트)
#             try:
#                 np.save(self._sem_save_path_npy, semantic_image_once)
#             except Exception as e:
#                 print(f"[WARN] semantic NPY save failed: {e}")

#             self._sem_saved_once = True

#         cost_map = np.ones_like(semantic_image, dtype=np.uint8) * 5
#         cost_map[semantic_image == 7] = 0
#         cost_map[semantic_image == 14] = 0
#         cost_map[semantic_image == 8] = 155
#         cost_map[semantic_image == 6] = 255
#         cost_map[semantic_image == 10] = 255
#         cost_map[semantic_image == 4] = 255
#         self.bev_obstacle_map = cost_map

#         ego_bb = self.player.bounding_box
#         ego_transform = self.player.get_transform()
#         vertices = [v for v in ego_bb.get_world_vertices(ego_transform)]
#         bottom_vertices = sorted(vertices, key=lambda v: v.z)[:4]
#         ego_footprint_pixels = [world_to_bev_pixels(v, ego_transform) for v in bottom_vertices]
        
#         pixel_array = np.array(ego_footprint_pixels)
#         centroid = np.mean(pixel_array, axis=0)
#         sorted_pixels = sorted(ego_footprint_pixels, key=lambda p: math.atan2(p[1] - centroid[1], p[0] - centroid[0]))
#         cv2.fillPoly(self.bev_obstacle_map, [np.array(sorted_pixels, dtype=np.int32)], 0)

#         obstacle_pixels = np.argwhere(self.bev_obstacle_map >= 200)
#         downsampled_pixels = obstacle_pixels[::5]
#         new_ox, new_oy = self.bev_pixels_to_world(downsampled_pixels)
#         self.obstacle_ox = new_ox
#         self.obstacle_oy = new_oy
#         return True
    
#     def bev_pixels_to_world(self, pixels):
#         ego_transform = self.player.get_transform()
#         ego_to_world_matrix = np.array(ego_transform.get_matrix())
#         ox, oy = [], []
#         for p in pixels:
#             local_y = (p[1] - BEV_SHAPE[1] / 2) * BEV_Y_BOUND[2]
#             local_x = (BEV_SHAPE[0] / 2 - p[0]) * BEV_X_BOUND[2]
#             local_p = np.array([local_x, local_y, 0.0, 1.0])
#             world_p = ego_to_world_matrix @ local_p
#             ox.append(world_p[0])
#             oy.append(world_p[1])
#         return ox, oy
 
#     def spawn_static_vehicles(self):
#         """
#         미리 정의된 모든 주차 공간에 40% 확률로 장애물 차량을
#         무작위로 정적 스폰합니다.
#         """
#         # --- 1. Town04의 모든 주차 공간 위치를 Transform 리스트로 정의 ---
#         ALL_PARKING_SPOTS = [
#                 # row 1
#             carla.Transform(carla.Location(x=298.5, y=-235.73, z=0.3), carla.Rotation(yaw=0.0)),  # 1-1
#             carla.Transform(carla.Location(x=298.5, y=-232.73, z=0.3), carla.Rotation(yaw=0.0)),  # 1-2
#             carla.Transform(carla.Location(x=298.5, y=-229.53, z=0.3), carla.Rotation(yaw=0.0)),  # 1-3
#             carla.Transform(carla.Location(x=298.5, y=-226.43, z=0.3), carla.Rotation(yaw=0.0)),  # 1-4
#             carla.Transform(carla.Location(x=298.5, y=-223.43, z=0.3), carla.Rotation(yaw=0.0)),  # 1-5
#             carla.Transform(carla.Location(x=298.5, y=-220.23, z=0.3), carla.Rotation(yaw=0.0)),  # 1-6
#             carla.Transform(carla.Location(x=298.5, y=-217.23, z=0.3), carla.Rotation(yaw=0.0)),  # 1-7
#             carla.Transform(carla.Location(x=298.5, y=-214.03, z=0.3), carla.Rotation(yaw=0.0)),  # 1-8
#             carla.Transform(carla.Location(x=298.5, y=-210.73, z=0.3), carla.Rotation(yaw=0.0)),  # 1-9
#             carla.Transform(carla.Location(x=298.5, y=-207.30, z=0.3), carla.Rotation(yaw=0.0)), # 1-10
#             carla.Transform(carla.Location(x=298.5, y=-204.23, z=0.3), carla.Rotation(yaw=0.0)), # 1-11
#             carla.Transform(carla.Location(x=298.5, y=-201.03, z=0.3), carla.Rotation(yaw=0.0)), # 1-12
#             carla.Transform(carla.Location(x=298.5, y=-198.03, z=0.3), carla.Rotation(yaw=0.0)), # 1-13
#             carla.Transform(carla.Location(x=298.5, y=-194.90, z=0.3), carla.Rotation(yaw=0.0)), # 1-14
#             carla.Transform(carla.Location(x=298.5, y=-191.53, z=0.3), carla.Rotation(yaw=0.0)), # 1-15
#             carla.Transform(carla.Location(x=298.5, y=-188.20, z=0.3), carla.Rotation(yaw=0.0)), # 1-16

#             # row 2
#             carla.Transform(carla.Location(x=290.9, y=-235.73, z=0.3), carla.Rotation(yaw=180.0)), # 2-1
#             carla.Transform(carla.Location(x=290.9, y=-232.73, z=0.3), carla.Rotation(yaw=180.0)), # 2-2
#             carla.Transform(carla.Location(x=290.9, y=-229.53, z=0.3), carla.Rotation(yaw=180.0)), # 2-3
#             carla.Transform(carla.Location(x=290.9, y=-226.43, z=0.3), carla.Rotation(yaw=180.0)), # 2-4
#             carla.Transform(carla.Location(x=290.9, y=-223.43, z=0.3), carla.Rotation(yaw=180.0)), # 2-5
#             carla.Transform(carla.Location(x=290.9, y=-220.23, z=0.3), carla.Rotation(yaw=180.0)), # 2-6
#             carla.Transform(carla.Location(x=290.9, y=-217.23, z=0.3), carla.Rotation(yaw=180.0)), # 2-7
#             carla.Transform(carla.Location(x=290.9, y=-214.03, z=0.3), carla.Rotation(yaw=180.0)), # 2-8
#             carla.Transform(carla.Location(x=290.9, y=-210.73, z=0.3), carla.Rotation(yaw=180.0)), # 2-9
#             carla.Transform(carla.Location(x=290.9, y=-207.30, z=0.3), carla.Rotation(yaw=180.0)), # 2-10
#             carla.Transform(carla.Location(x=290.9, y=-204.23, z=0.3), carla.Rotation(yaw=180.0)), # 2-11
#             carla.Transform(carla.Location(x=290.9, y=-201.03, z=0.3), carla.Rotation(yaw=180.0)), # 2-12
#             carla.Transform(carla.Location(x=290.9, y=-198.03, z=0.3), carla.Rotation(yaw=180.0)), # 2-13
#             carla.Transform(carla.Location(x=290.9, y=-194.90, z=0.3), carla.Rotation(yaw=180.0)), # 2-14
#             carla.Transform(carla.Location(x=290.9, y=-191.53, z=0.3), carla.Rotation(yaw=180.0)), # 2-15
#             carla.Transform(carla.Location(x=290.9, y=-188.20, z=0.3), carla.Rotation(yaw=180.0)), # 2-16

#             # row 3
#             carla.Transform(carla.Location(x=280.0, y=-235.73, z=0.3), carla.Rotation(yaw=0.0)),  # 3-1
#             carla.Transform(carla.Location(x=280.0, y=-232.73, z=0.3), carla.Rotation(yaw=0.0)),   # 3-2
#             carla.Transform(carla.Location(x=280.0, y=-229.53, z=0.3), carla.Rotation(yaw=0.0)), # 3-3
#             carla.Transform(carla.Location(x=280.0, y=-226.43, z=0.3), carla.Rotation(yaw=0.0)), # 3-4
#             carla.Transform(carla.Location(x=280.0, y=-223.43, z=0.3), carla.Rotation(yaw=0.0)), # 3-5
#             carla.Transform(carla.Location(x=280.0, y=-220.23, z=0.3), carla.Rotation(yaw=0.0)), # 3-6
#             carla.Transform(carla.Location(x=280.0, y=-217.23, z=0.3), carla.Rotation(yaw=0.0)), # 3-7
#             carla.Transform(carla.Location(x=280.0, y=-214.03, z=0.3), carla.Rotation(yaw=0.0)), # 3-8
#             carla.Transform(carla.Location(x=280.0, y=-210.73, z=0.3), carla.Rotation(yaw=0.0)), # 3-9
#             carla.Transform(carla.Location(x=280.0, y=-207.30, z=0.3), carla.Rotation(yaw=0.0)), # 3-10
#             carla.Transform(carla.Location(x=280.0, y=-204.23, z=0.3), carla.Rotation(yaw=0.0)), # 3-11
#             carla.Transform(carla.Location(x=280.0, y=-201.03, z=0.3), carla.Rotation(yaw=0.0)), # 3-12
#             carla.Transform(carla.Location(x=280.0, y=-198.03, z=0.3), carla.Rotation(yaw=0.0)), # 3-13
#             carla.Transform(carla.Location(x=280.0, y=-194.90, z=0.3), carla.Rotation(yaw=0.0)), # 3-14
#             carla.Transform(carla.Location(x=280.0, y=-191.53, z=0.3), carla.Rotation(yaw=0.0)), # 3-15
#             carla.Transform(carla.Location(x=280.0, y=-188.20, z=0.3), carla.Rotation(yaw=0.0)), # 3-16

#             # row 4
#             carla.Transform(carla.Location(x=272.5, y=-235.73, z=0.3), carla.Rotation(yaw=180.0)),  # 4-1
#             carla.Transform(carla.Location(x=272.5, y=-232.73, z=0.3), carla.Rotation(yaw=180.0)),  # 4-2
#             carla.Transform(carla.Location(x=272.5, y=-229.53, z=0.3), carla.Rotation(yaw=180.0)),  # 4-3
#             carla.Transform(carla.Location(x=272.5, y=-226.43, z=0.3), carla.Rotation(yaw=180.0)),  # 4-4
#             carla.Transform(carla.Location(x=272.5, y=-223.43, z=0.3), carla.Rotation(yaw=180.0)), # 4-5
#             carla.Transform(carla.Location(x=272.5, y=-220.23, z=0.3), carla.Rotation(yaw=180.0)), # 4-6
#             carla.Transform(carla.Location(x=272.5, y=-217.23, z=0.3), carla.Rotation(yaw=180.0)), # 4-7
#             carla.Transform(carla.Location(x=272.5, y=-214.03, z=0.3), carla.Rotation(yaw=180.0)), # 4-8
#             carla.Transform(carla.Location(x=272.5, y=-210.73, z=0.3), carla.Rotation(yaw=180.0)), # 4-9
#             carla.Transform(carla.Location(x=272.5, y=-207.30, z=0.3), carla.Rotation(yaw=180.0)), # 4-10
#             carla.Transform(carla.Location(x=272.5, y=-204.23, z=0.3), carla.Rotation(yaw=180.0)), # 4-11
#             carla.Transform(carla.Location(x=272.5, y=-201.03, z=0.3), carla.Rotation(yaw=180.0)), # 4-12
#             carla.Transform(carla.Location(x=272.5, y=-198.03, z=0.3), carla.Rotation(yaw=180.0)), # 4-13
#             carla.Transform(carla.Location(x=272.5, y=-194.90, z=0.3), carla.Rotation(yaw=180.0)), # 4-14
#             carla.Transform(carla.Location(x=272.5, y=-191.53, z=0.3), carla.Rotation(yaw=180.0)), # 4-15
#             carla.Transform(carla.Location(x=272.5, y=-188.20, z=0.3), carla.Rotation(yaw=180.0)), # 4-16
#         ]

#         # --- 2. 차량 필터링 ---
#         blacklist = [
#             'vehicle.carlamotors.carlacola', 'vehicle.tesla.cybertruck', 'vehicle.carlamotors.firetruck',
#             'vehicle.mercedes.sprinter', 'vehicle.ford.ambulance', 'vehicle.nissan.patrol_2021',
#             'vehicle.chevrolet.impala', 'vehicle.tesla.model3'
#         ]
#         all_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
#         blueprints = [bp for bp in all_blueprints
#                       if bp.id not in blacklist and int(bp.get_attribute('number_of_wheels')) == 4]
        
#         # --- 3. 90% 확률로 무작위 스폰 ---
#         for spot_transform in ALL_PARKING_SPOTS:
#             # 확률을 90%로 조정합니다.
#             if random.random() < 0.9:
#                 try:
#                     vehicle = self.world.try_spawn_actor(random.choice(blueprints), spot_transform)
#                     if vehicle:
#                         vehicle.set_simulate_physics(False)
#                         self.actor_list.append(vehicle)
#                 except Exception:
#                     pass
#         print(f"Spawned obstacle vehicles.")

#     def destroy(self):
#         print("Destroying actors...")
#         for actor in self.actor_list:
#             if actor and actor.is_alive:
#                 actor.destroy()
#         print("All actors destroyed.")

