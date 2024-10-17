'''
Пример использования достаточно грубой, но рабочей функции, которая начинает ехать на утку по коориднате -- но не очень точно, поскольку проблемы с точностью, то используем аналог LF, а когла близко, то это
'''


from gym_duckietown.tasks.task_solution import TaskSolution
import cv2
import numpy as np
import math


def calculate_control_commands(x_current, y_current, theta_current, x_target, y_target):
    delta_x = x_target - x_current
    delta_y = y_target - y_current
    target_angle = math.atan2(delta_y, delta_x)

    angle_error = target_angle - theta_current
    angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi  

    distance_to_target = math.sqrt(delta_x**2 + delta_y**2)

    dead_zone = 0.1  
    if abs(angle_error) < dead_zone:
        angular_velocity = 0
    else:
        angular_velocity = angle_error * 2  

    linear_velocity = min(distance_to_target, 0.2) 

    return linear_velocity, angular_velocity


class AdoQualificationTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def solve(self):
        env = self.generated_task['env']

        obs, _, _, _ = env.step([0,0]) 

        # получение координат утки, до которой необходимо доехать 
        goal = self.generated_task['target_coordinates']

        tmp_pos = env.cur_pos
        target_pos = [3.7, 0.7]
        tmp_angle = env.cur_angle  

        while True:
            linear_velocity, angular_velocity = calculate_control_commands(tmp_pos[0], tmp_pos[1], tmp_angle, target_pos[0], target_pos[1])
            # action = (linear_velocity, angular_velocity )
            action = (linear_velocity, angular_velocity - np.pi/2)

            print(f"Action: {action}")

            env.step(action)

            tmp_pos = env.cur_pos
            tmp_angle = env.cur_angle  

            distance_to_target = math.sqrt((target_pos[0] - tmp_pos[0])**2 + (target_pos[1] - tmp_pos[1])**2)
            if distance_to_target < 0.1:  
                print("Target reached!")
                break



        target_info = self.generated_task['start_target_info']
        print('\n\n\n', target_info, '\n\n\n')
        
if __name__ == "__main__":
    # код ниже требуется для возможности запуска вашего решения в описываемом образе, при отправки решения в систему проверки данный код не требуется
    from gym_duckietown.tasks.default.task_generator import DefaultTaskGenerator

    task_generator = DefaultTaskGenerator()
    task_generator.generate_task()
    solution = AdoQualificationTaskSolution(task_generator.generated_task)
    solution.solve()
