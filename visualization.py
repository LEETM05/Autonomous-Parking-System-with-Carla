# visualization.py

import matplotlib.pyplot as plt
import os
import cv2
import carla
from carla_utils import world_to_bev_pixels

def visualize_plan_debug(obstacle_map, goal_pixel, path, closed_set, ego_transform, frame_num):
    if frame_num == 1:
        if not os.path.exists('results'): os.makedirs('results')
        cv2.imwrite(f'results/bev_map_frame_{frame_num:04d}.png', obstacle_map)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(obstacle_map, cmap='gist_earth', origin='upper', vmin=0, vmax=255)

    if closed_set:
        explored_pixels = []
        for node in closed_set.values():
            node_loc = carla.Location(x=node.x[-1], y=node.y[-1])
            px, py = world_to_bev_pixels(node_loc, ego_transform)
            explored_pixels.append((px, py))
        if explored_pixels:
            exp_x, exp_y = zip(*explored_pixels)
            ax.scatter(exp_x, exp_y, c='orange', s=2, alpha=0.1, label='Explored Nodes')

    start_pixel = (obstacle_map.shape[1] // 2, obstacle_map.shape[0] // 2)
    ax.scatter(start_pixel[0], start_pixel[1], c='lime', s=100, label='Start (Ego)', marker='s', zorder=5)
    
    if goal_pixel:
        ax.scatter(goal_pixel[0], goal_pixel[1], c='red', s=100, label='Goal', marker='*', zorder=5)
        
    if path:
        path_pixels = [world_to_bev_pixels(carla.Location(x=px, y=py), ego_transform) for px, py in zip(path.x, path.y)]
        path_x, path_y = zip(*path_pixels)
        ax.plot(path_x, path_y, "b-", linewidth=2, label="Path", zorder=4)

    ax.set_title(f"Planner's BEV Map (Frame {frame_num})")
    ax.legend()
    
    if not os.path.exists('results'): os.makedirs('results')
    fig.savefig(f'results/plan_{frame_num:04d}.png')
    plt.close(fig)