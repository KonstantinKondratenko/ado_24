from gym_duckietown.tasks.task_solution import TaskSolution
import cv2
import numpy as np


class LineDetector:
    def __init__(
        self,
        image_height: int,
        image_width: int,
        lower_hsv: np.ndarray = np.array([20, 100, 100]),
        upper_hsv: np.ndarray = np.array([30, 255, 255]),
    ) -> None:
        self._image_height = image_height
        self._image_width = image_width

        self._lower_hsv_yellow = lower_hsv
        self._upper_hsv_yellow = upper_hsv

        self._writer = cv2.VideoWriter(
            "output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (image_width, image_height),
        )

    def detect(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        src_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Create mask for yellow
        mask = cv2.inRange(hsv_image, self._lower_hsv_yellow, self._upper_hsv_yellow)
        # USE FOR DEBUG
        cv2.imwrite("mask.png", mask)

        out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self._writer.write(out)

        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # if contours:
        #     largest_contour = max(contours, key=cv2.contourArea)
        #     M = cv2.moments(largest_contour)
        #     if M["m00"] != 0:
        #         cx = int(M["m10"] / M["m00"])
        #         cy = int(M["m01"] / M["m00"])
        #         return cx, cy


class AdoQualificationTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def solve(self):
        env = self.generated_task["env"]

        # получение первого изображения с бота
        obs, _, _, _ = env.step([0, 0])  # нормальное тангенсальное ускорение
        heigth, width = obs.shape[:2]
        line_detector = LineDetector(heigth, width)

        # получение координат утки, до которой необходимо доехать
        goal = self.generated_task["target_coordinates"]

        for i in range(200):
            obs, _, _, _ = env.step([0, 0.3])
            line_detector.detect(obs)

        # получение изначальных данных о расстоянии и угле до конечной точки
        target_info = self.generated_task["start_target_info"]
        # в дальнейшем для получения обновленной информации можно вызывать self.generated_task['get_dist_angle'](env, prev_pos), где prev_pos - предыдущая позиция робота


if __name__ == "__main__":
    # код ниже требуется для возможности запуска вашего решения в описываемом образе, при отправки решения в систему проверки данный код не требуется
    from gym_duckietown.tasks.default.task_generator import DefaultTaskGenerator

    task_generator = DefaultTaskGenerator()
    task_generator.generate_task()
    solution = AdoQualificationTaskSolution(task_generator.generated_task)
    solution.solve()
