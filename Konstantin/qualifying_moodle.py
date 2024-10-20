from gym_duckietown.tasks.task_solution import TaskSolution
import cv2
import numpy as np
import math
from itertools import count

def get_yellow_moments(image, lower_hsv_yellow=np.array([20, 100, 100]), upper_hsv_yellow=np.array([30, 255, 255])):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv_yellow, upper_hsv_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
    return None, None

def near_duck(cur: list, duck: list, target_dist: float = 0.2) -> bool:
    return math.sqrt((cur[0] - duck[0])**2 + (cur[1] - duck[1])**2) < target_dist

class AdoQualificationTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def solve(self):
        env = self.generated_task["env"]

        obs, _, _, _ = env.step([0, 0])
        obs = obs[obs.shape[0] // 3:, :]
        height, width = obs.shape[:2]
        goal = self.generated_task["target_coordinates"]

        def calculate_steering_angle(cx, image_width):
            center_x = image_width / 2
            deviation = cx - center_x
            steering_angle = deviation / center_x
            return steering_angle

        for i in count():
            cx, _ = get_yellow_moments(obs)
            if cx is not None:
                steering_angle = calculate_steering_angle(cx, width)
                obs, _, _, _ = env.step([0.2, -steering_angle])
            else:
                obs, _, _, _ = env.step([0, 0.3])

            obs = obs[obs.shape[0] // 3:, :]

            print(f'Step : {i} \t Curent position = {env.cur_pos[::2]} \t Target position = {goal} \t distance = {math.sqrt((env.cur_pos[::2][0] - goal[0])**2 + (env.cur_pos[::2][1] - goal[1])**2)}')

            if near_duck(env.cur_pos[::2], goal) or i > 2000:
                print('Stoping coz distance = ', math.sqrt((env.cur_pos[::2][0] - goal[0])**2 + (env.cur_pos[::2][1] - goal[1])**2))
                break
