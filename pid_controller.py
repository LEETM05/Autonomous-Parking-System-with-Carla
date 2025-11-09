# controllers.py

import math
import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd, integral_max=10.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral_max = integral_max
        self.clear()
    
    def clear(self):
        self._integral = 0
        self._previous_error = 0

    def update(self, error):
        self._integral += error
        self._integral = max(min(self._integral, self.integral_max), -self.integral_max)
        derivative = error - self._previous_error
        self._previous_error = error
        return self.Kp * error + self.Ki * self._integral + self.Kd * derivative

class DebugPurePursuitController:
    def __init__(self, wheelbase, max_steer_angle_deg):
        self.wheelbase = wheelbase
        self.max_steer_rad = math.radians(max_steer_angle_deg)

    def update(self, ego_transform, path, is_reverse, lookahead_distance, current_path_index):
        lookahead_idx = current_path_index
        start_location = path[current_path_index]['location']
        while lookahead_idx < len(path) - 1:
            dist_to_lookahead = start_location.distance(path[lookahead_idx]['location'])
            if dist_to_lookahead >= lookahead_distance:
                break
            lookahead_idx += 1
        target_waypoint = path[lookahead_idx]['location']

        target_vec = np.array([target_waypoint.x, target_waypoint.y, target_waypoint.z])
        world_to_vehicle_matrix = np.array(ego_transform.get_inverse_matrix())
        vehicle_frame_vec = world_to_vehicle_matrix @ np.append(target_vec, 1.0)
        
        local_x = vehicle_frame_vec[0]
        local_y = -vehicle_frame_vec[1]

        actual_lookahead_dist = math.hypot(local_x, local_y)
        if actual_lookahead_dist < 0.1: actual_lookahead_dist = 0.1

        alpha = math.atan2(local_y, local_x)
        steer_rad = math.atan2(2 * self.wheelbase * math.sin(alpha), actual_lookahead_dist)
        
        steer_normalized = steer_rad / self.max_steer_rad
        steer_output = np.clip(steer_normalized, -1.0, 1.0)
            
        return steer_output