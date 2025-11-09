import carla
import os
import queue
import random
import pygame # Pygame for keyboard control
import numpy as np
import math

# --- (Settings are the same as before) ---
HOST = '127.0.0.1'
PORT = 2000
SAVE_DISTANCE_THRESHOLD = 0.5
OUTPUT_FOLDER = 'bev_dataset_manual'

# 💡 [핵심 수정] run_carla_parking.py와 완벽히 동일한 BEV 설정을 가져옵니다.
BEV_RESOLUTION = 0.2
BEV_X_BOUND = [-10.0, 10.0, BEV_RESOLUTION]
BEV_Y_BOUND = [-10.0, 10.0, BEV_RESOLUTION]
IMAGE_WIDTH = int((BEV_X_BOUND[1] - BEV_X_BOUND[0]) / BEV_X_BOUND[2])  # 100
IMAGE_HEIGHT = int((BEV_Y_BOUND[1] - BEV_Y_BOUND[0]) / BEV_Y_BOUND[2]) # 100

# 💡 딥러닝 학습을 위해 해상도를 높입니다 (예: 8배). 비율은 그대로 유지됩니다.
UPSCALE_FACTOR = 8
HIGH_RES_WIDTH = IMAGE_WIDTH * UPSCALE_FACTOR   # 800
HIGH_RES_HEIGHT = IMAGE_HEIGHT * UPSCALE_FACTOR # 800

PARKING_SPOT_TRANSFORMS = [
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

def game_loop():
    pygame.init()
    actor_list = []
    try:
        # --- Pygame window setup ---
        display = pygame.display.set_mode((HIGH_RES_WIDTH, HIGH_RES_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('CARLA Manual Control')
        clock = pygame.time.Clock()

        # --- CARLA setup ---
        if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
        client = carla.Client(HOST, PORT)
        client.set_timeout(10.0)
        world = client.load_world('Town04_Opt')
        # world.set_weather(carla.WeatherParameters.CloudySunset)
        
        # --- 동기 모드 설정 ---
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05 # 20 FPS
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # 1. Ego Vehicle 스폰
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        # 💡 [수정 1] 주차장 앞 도로에 차량을 직접 스폰합니다.
        spawn_transform = carla.Transform(
            carla.Location(x=285.0, y=-208.0, z=0.3), 
            carla.Rotation(yaw=270.0)
        )
        # Ego Vehicle 스폰
        vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
        actor_list.append(vehicle)

        # 💡 --- [핵심 수정] 필터링을 적용하여 장애물 차량 스폰 ---
        # 1. 블랙리스트 정의: 너무 크거나 부적합한 차량 제외
        blacklist = [
            'vehicle.carlamotors.carlacola', 'vehicle.tesla.cybertruck',
            'vehicle.carlamotors.firetruck', 'vehicle.mercedes.sprinter',
            'vehicle.ford.ambulance', 'vehicle.nissan.patrol_2021',
            'vehicle.chevrolet.impala',
            'vehicle.tesla.model3'  # 데이터 수집용 차량(Ego Vehicle)도 제외
        ]
        
        # 2. 모든 차량 블루프린트를 가져온 후 필터링
        all_blueprints = blueprint_library.filter('vehicle.*')
        background_vehicle_bps = [bp for bp in all_blueprints
                                  if bp.id not in blacklist and
                                  int(bp.get_attribute('number_of_wheels')) == 4]
        
        for spot_transform in PARKING_SPOT_TRANSFORMS:
            if random.random() < 0.4:  # 40% 확률로 해당 공간에 차량 생성
                bg_bp = random.choice(background_vehicle_bps)
                # try...except 블록으로 스폰 실패 시에도 프로그램이 중단되지 않도록 함
                try:
                    bg_vehicle = world.try_spawn_actor(bg_bp, spot_transform)
                    if bg_vehicle:
                        actor_list.append(bg_vehicle)
                except RuntimeError as e:
                    print(f"Warning: Could not spawn background vehicle at {spot_transform.location}. Reason: {e}")
        # --------------------------------------------------------------------------
        # 3. BEV 카메라 설정
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(HIGH_RES_WIDTH))
        cam_bp.set_attribute('image_size_y', str(HIGH_RES_HEIGHT))
        # -------------------------------------------
        
       # z=40 높이에서 20m 너비를 정확히 보기 위한 FOV를 수학적으로 계산
        # 이 공식은 run_carla_parking.py의 내부 원리와 동일합니다.
        cam_height = 40 
        fov = str(2 * math.degrees(math.atan2(BEV_X_BOUND[1], cam_height)))
        cam_bp.set_attribute('fov', fov)
        
        cam_transform = carla.Transform(carla.Location(z=cam_height), carla.Rotation(pitch=-90))
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        actor_list.append(camera)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        # --- Main Loop ---
        print("Starting manual data collection. Use WASD to drive. Press Esc to stop.")
        last_saved_location = vehicle.get_location()
        frame_count = 0
        
        world.tick()

        while True:
            # --- Event Handling ---
            image = image_queue.get()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE):
                    return # Exit loop

            # --- Keyboard Controls ---
            keys = pygame.key.get_pressed()
            steer = -0.8 if keys[pygame.K_a] else (0.8 if keys[pygame.K_d] else 0.0)
            throttle = 0.0
            reverse = False
            brake = 0.0

            # 💡 'W'는 전진, 'S'는 후진으로 명확히 분리합니다.
            if keys[pygame.K_w]:
                throttle = 0.6  # 약간의 속도 조절
                reverse = False
            elif keys[pygame.K_s]:
                throttle = 0.6  # 후진 시에도 throttle 사용
                reverse = True
            else:
                # 아무 키도 누르지 않으면 브레이크를 살짝 밟아 차가 흐르지 않게 함
                brake = 0.1

            # 스페이스바를 브레이크로 사용할 수 있도록 추가
            if keys[pygame.K_SPACE]:
                throttle = 0.0
                brake = 1.0

            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, reverse=reverse))

            # --- Image Saving Logic ---
            world.tick()
            clock.tick_busy_loop(30) # Limit to 30 FPS
            
            # Display image on pygame window
            try:
                image = image_queue.get(block=False)
            except queue.Empty:
                continue # 이미지가 없으면 이번 프레임은 건너뜀
            # 💡 [수정] 이미지 데이터를 고해상도에 맞게 변환합니다.
            array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((HIGH_RES_HEIGHT, HIGH_RES_WIDTH, 4))
            surface = pygame.surfarray.make_surface(array[:, :, :3].swapaxes(0, 1))
            display.blit(surface, (0, 0))
            pygame.display.flip()
            
            # Save image if moved enough
            current_location = vehicle.get_location()
            if current_location.distance(last_saved_location) >= SAVE_DISTANCE_THRESHOLD:
                filename = os.path.join(OUTPUT_FOLDER, f'frame_{frame_count:05d}.png')
                image.save_to_disk(filename)
                print(f"Saved {filename}")
                last_saved_location = current_location
                frame_count += 1
                
            world.tick()

    finally:
        print("Cleaning up actors and restoring settings...")
        if 'client' in locals() and actor_list:
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        # 원래 설정으로 복원
        if 'original_settings' in locals():
            world.apply_settings(original_settings)
        pygame.quit()
        print("Done.")

if __name__ == '__main__':
    game_loop()