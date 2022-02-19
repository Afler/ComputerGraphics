from PIL import Image
import numpy as np
from typing import Optional, Tuple, Union


class OBJ3DModel:
    def __init__(self):
        self.img_arr: Optional[np.ndarray] = None

    def read_model(self, path: str):
        pass

    def arr_init(self):
        self.img_arr = np.zeros((self.height, self.width,
                                 self.channels), dtype=np.uint8)


class MyImage:
    def __init__(self):
        self.img_arr: Optional[np.ndarray] = None
        self.width: int = 0
        self.height: int = 0
        self.channels: int = 3
        self.delta_t: float = 0.01
        self.obj3D = OBJ3DModel()

    # инициализация массива методом библиотеки numpy
    def arr_init(self):
        self.img_arr = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)

    # установка значения цвета пиксела кортежем (tuple) из трех значений R, G, B или одним значением,
    # если изображение одноканальное (полутоновое)
    def set_pixel(self, x: int, y: int, color: Union[Tuple[int, int, int], int] = (0, 0, 0)):
        self.img_arr[y, x, :] = color

    # конвертация массива в объект класса Image библиотеки Pillow и вывод на экран
    # см. https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.show
    def imshow(self):
        Image.fromarray(self.img_arr, 'RGB').show()

    # конвертация массива в объект класса Image библиотеки Pillow и сохранение его
    # см. https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
    def save(self, path: str):
        im = Image.fromarray(self.img_arr, 'RGB')
        im.save()

    # рисование линии, первый вариант алгоритма
    def draw_line_v1(self, x0: int, y0: int, x1: int, y1: int, color: Union[Tuple[int, int, int], int]):
        for t in np.arange(0, 1, self.delta_t):
            x = int(x0 * (1.0 - t) + x1 * t)
            y = int(y0 * (1.0 - t) + y1 * t)
            self.set_pixel(x, y, color)

    # рисование линии, второй вариант алгоритма
    def draw_line_v2(self, x0: int, y0: int, x1: int, y1: int, color: Union[Tuple[int, int, int], int]):
        for x in range(int(x0), int(x1), 1):
            t = (x - x0) / (float)(x1 - x0)
            y = int(y0 * (1.0 - t) + y1 * t)
            self.set_pixel(int(x), int(y), color)
            sleep = True

    # рисование линии, третий вариант алгоритма
    def draw_line_v3(self, x0: int, y0: int, x1: int, y1: int, color: Union[Tuple[int, int, int], int]):
        steep = False
        if np.abs(x0 - x1) < np.abs(y0 - y1):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            steep = True
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        for x in range(int(x0), int(x1), 1):
            t = (x - x0) / (float)(x1 - x0)
            y = int(y0 * (1.0 - t) + y1 * t)
            if steep:
                self.set_pixel(int(y), int(x), color)
            else:
                self.set_pixel(int(x), int(y), color)

    # рисование линии, четвертый вариант алгоритма (алгоримтм Брезенхема)
    def draw_line_v4(self, x0: int, y0: int, x1: int, y1: int, color: Union[Tuple[int, int, int], int]):
        steep = False
        if np.abs(x0 - x1) < np.abs(y0 - y1):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            steep = True
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        dx = x1 - x0
        dy = y1 - y0
        derror = np.abs(dy / (float)(dx))
        error = 0
        y = y0
        for x in range(int(x0), int(x1), 1):
            if steep:
                self.set_pixel(int(y), int(x), color)
            else:
                self.set_pixel(int(x), int(y), color)
            error += derror
            if error > 0.5:
                y += 1 if y1 > y0 else -1
                error -= 1

    # в качестве параметра можно передать саму функцию отрисовки линии
    def draw_star(self, draw_line_variant):
        for i in range(13):
            alpha = 2 * np.pi * i / 13
            draw_line_variant(100, 100, 100 + 95 * np.cos(alpha), 100 + 95 * np.sin(alpha), (255, 255, 255))

    # отрисовка вершин считанной 3D модели
    def draw_vertices(self):
        pass

    # отрисовка вершин
    def draw_edges(self):
        pass


if __name__ == "__main__":
    print("hi guys")
    w = 128
    h = 128
    arr = np.zeros((h, w), dtype=np.uint8)
    image = Image.fromarray(arr, 'L')
    # image.show()
    arr2 = np.copy(arr)
    arr2 += 255
    image2 = Image.fromarray(arr2, 'L')
    # image2.show()
    arr3 = np.zeros((h, w, 3), dtype=np.uint8)
    arr3[:, :, :] = (255, 0, 0)
    image3 = Image.fromarray(arr3, 'RGB')
    # image3.show()
    arr4 = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            for k in range(3):
                arr4[i, j, k] = (i + j + k) % 256
    image4 = Image.fromarray(arr4, 'RGB')
    # image4.show()
    img = MyImage()
    img.width = 900
    img.height = 900
    img.arr_init()
    # img.draw_star(img.draw_line_v1)
    img.draw_line_v1(125, 0, 125, 225, (255, 255, 255))
    img.imshow()
