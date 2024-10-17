from gym_duckietown.tasks.task_solution import TaskSolution
import cv2
import numpy as np


class SegmentYellow:
    def __init__(self, frame_height, frame_width):
        self.writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))
    
    def segment(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        src = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("src.png", src)
        
        # Определение диапазона желтого цвета в HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        # Создание маски для желтого цвета
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        cv2.imwrite("mask.png", mask)
        out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self.writer.write(out)
        
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # if contours:
        #     largest_contour = max(contours, key=cv2.contourArea)
        #     M = cv2.moments(largest_contour)
        #     if M["m00"] != 0:
        #         cx = int(M["m10"] / M["m00"])
        #         cy = int(M["m01"] / M["m00"])
        #         return cx, cy
            
        
        # for contour in contours:
        #     M = cv2.moments(contour)
            
        #     # Вычисление центра масс
        #     cx = int(M['m10'] / M['m00'])
        #     cy = int(M['m01'] / M['m00'])

        #     # Рисование центра масс на изображении
        #     cv2.circle(src, (cx, cy), 5, (0, 255, 0), -1)



class AdoQualificationTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def solve(self):
        env = self.generated_task['env']

        # получение первого изображения с бота
        obs, _, _, _ = env.step([0,0]) # нормальное тангенсальное ускорение
        segmentator = SegmentYellow(obs.shape[0], obs.shape[1])

        # получение координат утки, до которой необходимо доехать 
        goal = self.generated_task['target_coordinates']
        
        for i in range(200):
            obs, _, _, _ = env.step([0, 0.3])  
            segmentator.segment(obs)
        
        # получение изначальных данных о расстоянии и угле до конечной точки
        target_info = self.generated_task['start_target_info']
        # в дальнейшем для получения обновленной информации можно вызывать self.generated_task['get_dist_angle'](env, prev_pos), где prev_pos - предыдущая позиция робота
        
if __name__ == "__main__":
    # код ниже требуется для возможности запуска вашего решения в описываемом образе, при отправки решения в систему проверки данный код не требуется
    from gym_duckietown.tasks.default.task_generator import DefaultTaskGenerator

    task_generator = DefaultTaskGenerator()
    task_generator.generate_task()
    solution = AdoQualificationTaskSolution(task_generator.generated_task)
    solution.solve()