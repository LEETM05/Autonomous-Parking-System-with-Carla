import matplotlib.pyplot as plt
import numpy as np
import math

import cv2
import os, sys
import carla
from carla_utils import world_to_bev_pixels

PI = np.pi


class Arrow:
    def __init__(self, x, y, theta, L, c):
        angle = np.deg2rad(30)
        d = 0.3 * L
        w = 2

        x_start = x
        y_start = y
        x_end = x + L * np.cos(theta)
        y_end = y + L * np.sin(theta)

        theta_hat_L = theta + PI - angle
        theta_hat_R = theta + PI + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        plt.plot([x_start, x_end], [y_start, y_end], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_L],
                 [y_hat_start, y_hat_end_L], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_R],
                 [y_hat_start, y_hat_end_R], color=c, linewidth=w)


class Car:
    def __init__(self, x, y, yaw, w, L):
        theta_B = PI + yaw

        xB = x + L / 4 * np.cos(theta_B)
        yB = y + L / 4 * np.sin(theta_B)

        theta_BL = theta_B + PI / 2
        theta_BR = theta_B - PI / 2

        x_BL = xB + w / 2 * np.cos(theta_BL)        # Bottom-Left vertex
        y_BL = yB + w / 2 * np.sin(theta_BL)
        x_BR = xB + w / 2 * np.cos(theta_BR)        # Bottom-Right vertex
        y_BR = yB + w / 2 * np.sin(theta_BR)

        x_FL = x_BL + L * np.cos(yaw)               # Front-Left vertex
        y_FL = y_BL + L * np.sin(yaw)
        x_FR = x_BR + L * np.cos(yaw)               # Front-Right vertex
        y_FR = y_BR + L * np.sin(yaw)

        plt.plot([x_BL, x_BR, x_FR, x_FL, x_BL],
                 [y_BL, y_BR, y_FR, y_FL, y_BL],
                 linewidth=1, color='black')

        Arrow(x, y, yaw, L / 2, 'black')
        # plt.axis("equal")
        # plt.show()


def draw_car(x, y, yaw, steer, C, color='black'):
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
    Arrow(x, y, yaw, C.WB * 0.8, color)

def visualize_plan_debug(obstacle_map, goal_pixel, path, closed_set, ego_transform, frame_num):
    """플래너의 BEV 맵과 경로를 시각화하고 파일로 저장합니다."""
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
        path_pixels = []
        for i in range(len(path.x)):
            path_loc = carla.Location(x=path.x[i], y=path.y[i])
            px, py = world_to_bev_pixels(path_loc, ego_transform)
            path_pixels.append((px, py))
        
        path_x, path_y = zip(*path_pixels)
        ax.plot(path_x, path_y, "b-", linewidth=2, label="Path", zorder=4)

    ax.set_title(f"Planner's BEV Map (Frame {frame_num})")
    ax.legend()
    
    if not os.path.exists('results'): os.makedirs('results')
    fig.savefig(f'results/plan_{frame_num:04d}.png')
    plt.close(fig)


if __name__ == '__main__':
    # Arrow(-1, 2, 60)
    Car(0, 0, 1, 2, 60)
