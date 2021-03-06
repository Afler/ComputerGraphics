from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

from lr1.Parser import Parser


def calculateNormal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    nx = (y1 - y0) * (z1 - z2) - (z1 - z0) * (y1 - y2)
    ny = (x1 - x0) * (z1 - z2) - (z1 - z0) * (x1 - x2)
    nz = (x1 - x0) * (y1 - y2) - (y1 - y0) * (x1 - x2)
    return nx, ny, nz


def scalarProduct(x0, y0, z0, x1, y1, z1):
    res = (x0 * x1 + y0 * y1 + z0 * z1) / (np.sqrt(x0 * x0 + y0 * y0 + z0 * z0) * np.sqrt(x1 * x1 + y1 * y1 + z1 * z1))
    return res


def projectiveTransformation(x, y, z, ax, ay, u0, v0, tz):
    u = (ax * x + u0 * (z + tz)) / z
    v = (ay * y + v0 * (z + tz)) / z
    return u, v


class OBJ3DModel:
    def __init__(self):
        self.vertexList: Optional[np.ndarray] = None
        self.vertexListVN: Optional[np.ndarray] = None
        self.polyVertexIndexesList: Optional[np.ndarray] = None
        self.polyNormalIndexesList: Optional[np.ndarray] = None
        self.img = MyImage()

    def read_model(self, path: str):
        self.vertexList, self.vertexListVN = Parser.getVertexes(path)
        self.polyVertexIndexesList, self.polyNormalIndexesList = Parser.getPolygons(path)

    def arr_init(self):
        self.vertexList = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)

    def rotate(self, alpha, beta, gamma):
        alpha *= np.pi / 180
        beta *= np.pi / 180
        gamma *= np.pi / 180
        R = np.mat([[1.0, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]]) \
            * np.mat([[np.cos(beta), 0, np.sin(beta)], [0, 1.0, 0], [-np.sin(beta), 0, np.cos(beta)]]) \
            * np.mat([[np.cos(gamma), np.sin(gamma), 0], [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1.0]])

        coordinatesFinish = np.zeros((len(self.vertexList), 3), np.float64)
        for i in range(len(self.vertexList)):
            coordinatesFinish[i, 0] = self.vertexList[i].x
            coordinatesFinish[i, 1] = self.vertexList[i].y
            coordinatesFinish[i, 2] = self.vertexList[i].z
        coordinatesFinish = coordinatesFinish.dot(R).getA()
        for i in range(len(self.vertexList)):
            self.vertexList[i].x = coordinatesFinish[i, 0]
            self.vertexList[i].y = coordinatesFinish[i, 1]
            self.vertexList[i].z = coordinatesFinish[i, 2]

    def rotateWithNormal(self, alpha, beta, gamma):
        alpha *= np.pi / 180
        beta *= np.pi / 180
        gamma *= np.pi / 180
        R = np.mat([[1.0, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]]) \
            * np.mat([[np.cos(beta), 0, np.sin(beta)], [0, 1.0, 0], [-np.sin(beta), 0, np.cos(beta)]]) \
            * np.mat([[np.cos(gamma), np.sin(gamma), 0], [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1.0]])

        coordinatesFinish = np.zeros((len(self.vertexList), 3), np.float64)
        for i in range(len(self.vertexList)):
            coordinatesFinish[i, 0] = self.vertexList[i].x
            coordinatesFinish[i, 1] = self.vertexList[i].y
            coordinatesFinish[i, 2] = self.vertexList[i].z
        coordinatesFinish = coordinatesFinish.dot(R).getA()
        for i in range(len(self.vertexList)):
            self.vertexList[i].x = coordinatesFinish[i, 0]
            self.vertexList[i].y = coordinatesFinish[i, 1]
            self.vertexList[i].z = coordinatesFinish[i, 2]
        coordinatesFinish2 = np.zeros((len(self.vertexList), 3), np.float64)
        for i in range(len(self.vertexListVN)):
            coordinatesFinish2[i, 0] = self.vertexListVN[i].x
            coordinatesFinish2[i, 1] = self.vertexListVN[i].y
            coordinatesFinish2[i, 2] = self.vertexListVN[i].z
        coordinatesFinish2 = coordinatesFinish2.dot(R).getA()
        for i in range(len(self.vertexListVN)):
            self.vertexListVN[i].x = coordinatesFinish2[i, 0]
            self.vertexListVN[i].y = coordinatesFinish2[i, 1]
            self.vertexListVN[i].z = coordinatesFinish2[i, 2]

    def perspective(self, ax, ay, u0, v0, zsup):
        for i in range(len(self.vertexList)):
            self.vertexList[i].z += zsup
            self.vertexList[i].x = (self.vertexList[i].x * ax) / self.vertexList[i].z + u0
            self.vertexList[i].y = (self.vertexList[i].y * ay) / self.vertexList[i].z + v0
            self.vertexList[i].z -= zsup

    def centrilize(self):
        x_max = y_max = z_max = -20000000.0
        x_min = y_min = z_min = 20000000.0

        for i in range(len(self.vertexList)):
            if self.vertexList[i].x > x_max:
                x_max = self.vertexList[i].x
            if self.vertexList[i].y > y_max:
                y_max = self.vertexList[i].y
            if self.vertexList[i].z > z_max:
                z_max = self.vertexList[i].z
            if self.vertexList[i].x < x_min:
                x_min = self.vertexList[i].x
            if self.vertexList[i].y < y_min:
                y_min = self.vertexList[i].y
            if self.vertexList[i].z < z_min:
                z_min = self.vertexList[i].z

        x_min = x_max - (x_max - x_min) / 2
        y_min = y_max - (y_max - y_min) / 2
        z_min = z_max - (z_max - z_min) / 2

        for i in range(len(self.vertexList)):
            self.vertexList[i].x -= x_min
            self.vertexList[i].y -= y_min
            self.vertexList[i].z -= z_min

    def normalize(self):
        x = y = z = -20000000.0
        for i in range(len(self.vertexList)):
            if abs(self.vertexList[i].x > x):
                x = abs(self.vertexList[i].x)
            if abs(self.vertexList[i].y > y):
                y = abs(self.vertexList[i].y)
            if abs(self.vertexList[i].z > z):
                z = abs(self.vertexList[i].z)
        for i in range(len(self.vertexList)):
            self.vertexList[i].x /= x
            self.vertexList[i].y /= y
            self.vertexList[i].z /= z

    def draw_edges_v1(self, path: str, displacementX=0, displacementY=0, scaleX=1, scaleY=1):
        self.read_model(path)
        for vertexIndexes in self.polyVertexIndexesList:
            self.img.draw_line_v4(scaleX * self.vertexList[vertexIndexes[0] - 1].x - displacementX,
                                  scaleY * self.vertexList[vertexIndexes[0] - 1].y - displacementY,
                                  scaleX * self.vertexList[vertexIndexes[1] - 1].x - displacementX,
                                  scaleY * self.vertexList[vertexIndexes[1] - 1].y - displacementY,
                                  (255, 255, 255))
            self.img.draw_line_v4(scaleX * self.vertexList[vertexIndexes[1] - 1].x - displacementX,
                                  scaleY * self.vertexList[vertexIndexes[1] - 1].y - displacementY,
                                  scaleX * self.vertexList[vertexIndexes[2] - 1].x - displacementX,
                                  scaleY * self.vertexList[vertexIndexes[2] - 1].y, (255, 255, 255))
            self.img.draw_line_v4(scaleX * self.vertexList[vertexIndexes[2] - 1].x - displacementX,
                                  scaleY * self.vertexList[vertexIndexes[2] - 1].y - displacementY,
                                  scaleX * self.vertexList[vertexIndexes[0] - 1].x - displacementX,
                                  scaleY * self.vertexList[vertexIndexes[0] - 1].y - displacementY,
                                  (255, 255, 255))

    def draw_edges_v2(self, path: str, displacementX=0, displacementY=0, scaleX=1, scaleY=1):
        self.read_model(path)
        i = 1
        for vertexIndexes in self.polyVertexIndexesList:
            self.img.drawTriangle(scaleX * self.vertexList[vertexIndexes[0] - 1].x - displacementX,
                                  scaleY * self.vertexList[vertexIndexes[0] - 1].y - displacementY,
                                  scaleX * self.vertexList[vertexIndexes[1] - 1].x - displacementX,
                                  scaleY * self.vertexList[vertexIndexes[1] - 1].y - displacementY,
                                  scaleX * self.vertexList[vertexIndexes[2] - 1].x - displacementX,
                                  scaleY * self.vertexList[vertexIndexes[2] - 1].y - displacementY,
                                  (i % 255, (i + 50) % 255, (i + 100) % 255))
            i = i + 20

    def draw_edges_v3(self, path: str, displacementX=0, displacementY=0, scaleX=1, scaleY=1):
        self.read_model(path)
        i = 1
        for vertexIndexes in self.polyVertexIndexesList:
            nx, ny, nz = calculateNormal(self.vertexList[vertexIndexes[0] - 1].x,
                                         self.vertexList[vertexIndexes[0] - 1].y,
                                         self.vertexList[vertexIndexes[0] - 1].z,
                                         self.vertexList[vertexIndexes[1] - 1].x,
                                         self.vertexList[vertexIndexes[1] - 1].y,
                                         self.vertexList[vertexIndexes[1] - 1].z,
                                         self.vertexList[vertexIndexes[2] - 1].x,
                                         self.vertexList[vertexIndexes[2] - 1].y,
                                         self.vertexList[vertexIndexes[2] - 1].z)
            cosine = scalarProduct(nx, ny, nz, 0, 0, 1)
            if cosine < 0:
                self.img.drawTriangle_v2(scaleX * self.vertexList[vertexIndexes[0] - 1].x - displacementX,
                                         scaleY * self.vertexList[vertexIndexes[0] - 1].y - displacementY,
                                         self.vertexList[vertexIndexes[0] - 1].z,
                                         scaleX * self.vertexList[vertexIndexes[1] - 1].x - displacementX,
                                         scaleY * self.vertexList[vertexIndexes[1] - 1].y - displacementY,
                                         self.vertexList[vertexIndexes[1] - 1].z,
                                         scaleX * self.vertexList[vertexIndexes[2] - 1].x - displacementX,
                                         scaleY * self.vertexList[vertexIndexes[2] - 1].y - displacementY,
                                         self.vertexList[vertexIndexes[2] - 1].z,
                                         self.img.zBuffer, (-255 * cosine, 0, 0))

    def draw_projective_edges(self, path: str, displacementX=0, displacementY=0, scaleX=1, scaleY=1):
        self.read_model(path)
        self.centrilize()
        self.normalize()
        self.rotate(0, 90, 0)
        self.perspective(2000, 2000, 0, 0, 3)
        self.normalize()
        for vertexIndexes in self.polyVertexIndexesList:
            nx, ny, nz = calculateNormal(self.vertexList[vertexIndexes[0] - 1].x,
                                         self.vertexList[vertexIndexes[0] - 1].y,
                                         self.vertexList[vertexIndexes[0] - 1].z,
                                         self.vertexList[vertexIndexes[1] - 1].x,
                                         self.vertexList[vertexIndexes[1] - 1].y,
                                         self.vertexList[vertexIndexes[1] - 1].z,
                                         self.vertexList[vertexIndexes[2] - 1].x,
                                         self.vertexList[vertexIndexes[2] - 1].y,
                                         self.vertexList[vertexIndexes[2] - 1].z)
            cosine = scalarProduct(nx, ny, nz, 0, 0, 1)
            if cosine >= 0:
                self.img.drawTriangle_v2(scaleX * self.vertexList[vertexIndexes[0] - 1].x - displacementX,
                                         scaleY * self.vertexList[vertexIndexes[0] - 1].y - displacementY,
                                         scaleX * self.vertexList[vertexIndexes[0] - 1].z,
                                         scaleX * self.vertexList[vertexIndexes[1] - 1].x - displacementX,
                                         scaleY * self.vertexList[vertexIndexes[1] - 1].y - displacementY,
                                         scaleX * self.vertexList[vertexIndexes[1] - 1].z,
                                         scaleX * self.vertexList[vertexIndexes[2] - 1].x - displacementX,
                                         scaleY * self.vertexList[vertexIndexes[2] - 1].y - displacementY,
                                         scaleX * self.vertexList[vertexIndexes[2] - 1].z,
                                         self.img.zBuffer, (255 * cosine, 0, 0))

    def draw_guro_edges(self, path: str):
        self.read_model(path)
        self.centrilize()
        self.normalize()
        self.rotateWithNormal(0, 45, 0)
        self.perspective(-2000, -2000, -800, -800, 10)
        for i in range(len(self.polyVertexIndexesList)):
            vertexIndexes = self.polyVertexIndexesList[i]
            normalIndexes = self.polyNormalIndexesList[i]
            nx1 = self.vertexListVN[normalIndexes[0] - 1].x
            ny1 = self.vertexListVN[normalIndexes[0] - 1].y
            nz1 = self.vertexListVN[normalIndexes[0] - 1].z
            nx2 = self.vertexListVN[normalIndexes[1] - 1].x
            ny2 = self.vertexListVN[normalIndexes[1] - 1].y
            nz2 = self.vertexListVN[normalIndexes[1] - 1].z
            nx3 = self.vertexListVN[normalIndexes[2] - 1].x
            ny3 = self.vertexListVN[normalIndexes[2] - 1].y
            nz3 = self.vertexListVN[normalIndexes[2] - 1].z
            cosine1 = scalarProduct(nx1, ny1, nz1, 0, 0, 1)
            cosine2 = scalarProduct(nx2, ny2, nz2, 0, 0, 1)
            cosine3 = scalarProduct(nx3, ny3, nz3, 0, 0, 1)

            self.img.drawTriangle_v3(self.vertexList[vertexIndexes[0] - 1].x,
                                     self.vertexList[vertexIndexes[0] - 1].y,
                                     self.vertexList[vertexIndexes[0] - 1].z,
                                     self.vertexList[vertexIndexes[1] - 1].x,
                                     self.vertexList[vertexIndexes[1] - 1].y,
                                     self.vertexList[vertexIndexes[1] - 1].z,
                                     self.vertexList[vertexIndexes[2] - 1].x,
                                     self.vertexList[vertexIndexes[2] - 1].y,
                                     self.vertexList[vertexIndexes[2] - 1].z,
                                     self.img.zBuffer, cosine1, cosine2, cosine3)


class MyImage:
    def __init__(self):
        self.img_arr: Optional[np.ndarray] = None
        self.width: int = 0
        self.height: int = 0
        self.channels: int = 3
        self.delta_t: float = 0.01
        self.zBuffer: Optional[np.ndarray] = None

    # ?????????????????????????? z ????????????
    def zBuffer_init(self):
        self.zBuffer = np.zeros((self.height, self.width), dtype=np.float)
        self.zBuffer[:, :] = 30000

    # ?????????????????????????? ?????????????? ?????????????? ???????????????????? numpy
    def arr_init(self):
        self.img_arr = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)

    # ?????????????????? ???????????????? ?????????? ?????????????? ???????????????? (tuple) ???? ???????? ???????????????? R, G, B ?????? ?????????? ??????????????????,
    # ???????? ?????????????????????? ?????????????????????????? (??????????????????????)
    def set_pixel(self, x: int, y: int, color: Union[Tuple[int, int, int], int] = (0, 0, 0)):
        self.img_arr[y, x, :] = color

    # ?????????????????????? ?????????????? ?? ???????????? ???????????? Image ???????????????????? Pillow ?? ?????????? ???? ??????????
    # ????. https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.show
    def imshow(self):
        Image.fromarray(self.img_arr, 'RGB').show()

    # ?????????????????????? ?????????????? ?? ???????????? ???????????? Image ???????????????????? Pillow ?? ???????????????????? ??????
    # ????. https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
    def save(self, path):
        im = Image.fromarray(self.img_arr, 'RGB')
        im.save(path)

    # ?????????????????? ??????????, ???????????? ?????????????? ??????????????????
    def draw_line_v1(self, x0: int, y0: int, x1: int, y1: int, color: Union[Tuple[int, int, int], int]):
        for t in np.arange(0, 1, self.delta_t):
            x = int(x0 * (1.0 - t) + x1 * t)
            y = int(y0 * (1.0 - t) + y1 * t)
            self.set_pixel(x, y, color)

    # ?????????????????? ??????????, ???????????? ?????????????? ??????????????????
    def draw_line_v2(self, x0: int, y0: int, x1: int, y1: int, color: Union[Tuple[int, int, int], int]):
        for x in range(int(x0), int(x1), 1):
            t = (x - x0) / (float)(x1 - x0)
            y = int(y0 * (1.0 - t) + y1 * t)
            self.set_pixel(int(x), int(y), color)
            sleep = True

    # ?????????????????? ??????????, ???????????? ?????????????? ??????????????????
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

    # ?????????????????? ??????????, ?????????????????? ?????????????? ?????????????????? (???????????????? ????????????????????)
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
        derror = np.abs(dy / float(dx))
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

    # ?? ???????????????? ?????????????????? ?????????? ???????????????? ???????? ?????????????? ?????????????????? ??????????
    def draw_star(self, draw_line_variant):
        for i in range(13):
            alpha = 2 * np.pi * i / 13
            draw_line_variant(100, 100, 100 + 95 * np.cos(alpha), 100 + 95 * np.sin(alpha), (255, 255, 255))

    # ?????????????????? ???????????? ?????????????????? 3D ????????????
    def draw_vertices(self):
        pass

    # ?????????????????? ????????????
    def draw_edges(self):
        pass

    def drawTriangle(self, x0, y0, x1, y1, x2, y2, color: Tuple[int, int, int] = (255, 255, 255)):
        xmin = min(x0, x1, x2) if min(x0, x1, x2) < 0 else 0
        ymin = min(y0, y1, y2) if min(y0, y1, y2) < 0 else 0
        xmax = max(x0, x1, x2) if max(x0, x1, x2) < self.width else self.width
        ymax = max(y0, y1, y2) if max(y0, y1, y2) < self.height else self.height
        for xIndex in range(round(xmin), round(xmax)):
            for yIndex in range(round(ymin), round(ymax)):
                bar1, bar2, bar3 = convertToBarycentric(xIndex, yIndex, x0, y0, x1, y1, x2, y2)
                if bar1 > 0 and bar2 > 0 and bar3 > 0:
                    self.set_pixel(xIndex, yIndex, color)
        pass

    def drawTriangle_v2(self, x0, y0, z0, x1, y1, z1, x2, y2, z2, zBuffer,
                        color: Tuple[int, int, int] = (255, 255, 255)):
        xmin = min(x0, x1, x2) if min(x0, x1, x2) < 0 else 0
        ymin = min(y0, y1, y2) if min(y0, y1, y2) < 0 else 0
        xmax = max(x0, x1, x2) if max(x0, x1, x2) < self.width else self.width
        ymax = max(y0, y1, y2) if max(y0, y1, y2) < self.height else self.height
        if xmax < (-self.width):
            xmax = -self.width
        if ymax < (-self.height):
            ymax = -self.height
        if xmin < (-self.width):
            xmin = -self.width
        if ymin < (-self.height):
            ymin = -self.height
        for xIndex in range(int(xmin), int(xmax)):
            for yIndex in range(int(ymin), int(ymax)):
                if xIndex > 0:
                    break
                if yIndex > 0:
                    break

                bar1, bar2, bar3 = convertToBarycentric(xIndex, yIndex, x0, y0, x1, y1, x2, y2)
                if bar1 > 0 and bar2 > 0 and bar3 > 0:
                    sourceZ = bar1 * z0 + bar2 * z1 + bar3 * z2
                    if sourceZ < zBuffer[xIndex, yIndex]:
                        self.set_pixel(xIndex, yIndex, color)
                        zBuffer[xIndex, yIndex] = sourceZ
        pass

    def drawTriangle_v3(self, x0, y0, z0, x1, y1, z1, x2, y2, z2, zBuffer,
                        l1, l2, l3):
        xmin = min(x0, x1, x2) if min(x0, x1, x2) < 0 else 0
        ymin = min(y0, y1, y2) if min(y0, y1, y2) < 0 else 0
        xmax = max(x0, x1, x2) if max(x0, x1, x2) < self.width else self.width
        ymax = max(y0, y1, y2) if max(y0, y1, y2) < self.height else self.height
        if xmax < (-self.width):
            xmax = -self.width
        if ymax < (-self.height):
            ymax = -self.height
        if xmin < (-self.width):
            xmin = -self.width
        if ymin < (-self.height):
            ymin = -self.height
        for xIndex in range(int(xmin), int(xmax)):
            for yIndex in range(int(ymin), int(ymax)):
                if xIndex > 0:
                    break
                if yIndex > 0:
                    break

                bar1, bar2, bar3 = convertToBarycentric(xIndex, yIndex, x0, y0, x1, y1, x2, y2)
                colorSub = l1*bar1 + l2*bar2 + l3*bar3
                if colorSub <= 0:
                    if bar1 >= 0 and bar2 >= 0 and bar3 >= 0:
                        sourceZ = bar1 * z0 + bar2 * z1 + bar3 * z2
                        if sourceZ < zBuffer[xIndex, yIndex]:
                            self.set_pixel(xIndex, yIndex, (int(-colorSub * 255), 0, 0))
                            zBuffer[xIndex, yIndex] = sourceZ
        pass


def convertToBarycentric(x, y, x0: float, y0: float, x1: float, y1: float, x2: float, y2: float, eps=0.1):
    lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
    lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
    lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    return lambda0, lambda1, lambda2


def doLR1():
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
    img.width = 1000
    img.height = 1000
    img.arr_init()
    img.draw_star(img.draw_line_v1)
    img.save("lr1/1.jpg")
    img.draw_star(img.draw_line_v2)
    img.save("lr1/2.jpg")
    img.draw_star(img.draw_line_v3)
    img.save("lr1/3.jpg")
    img.draw_star(img.draw_line_v4)
    img.save("lr1/4.jpg")
    img.draw_line_v4(125, 0, 125, 225, (255, 255, 255))
    obj = OBJ3DModel()
    obj.img.width = 1600
    obj.img.height = 1600
    obj.img.arr_init()
    obj.draw_edges_v1("lr1/deer.obj", 650, 0, 1, -1)
    obj.img.save("lr1/deer.jpg")
    obj.img.imshow()


def doLR2():
    # drawTriangle(300, 500, 20, 30, 150, 700)
    obj = OBJ3DModel()
    obj.img.width = 1600
    obj.img.height = 1600
    obj.img.arr_init()
    # obj.draw_edges_v1("lr1/deer.obj", 650, 0, 1, -1)
    # obj.draw_edges_v2("lr1/deer.obj", 650, 0, 1, -1)
    # obj.img.save("lr2/deer_colored.jpg")
    obj.img.zBuffer_init()
    obj.draw_edges_v3("lr1/deer.obj", 650, 0, 1, -1)
    obj.img.save("lr2/deer_lighted_colored.jpg")
    obj.img.imshow()


def doLR3():
    # drawTriangle(300, 500, 20, 30, 150, 700)
    obj = OBJ3DModel()
    obj.img.width = 1600
    obj.img.height = 1600
    obj.img.arr_init()
    obj.img.zBuffer_init()
    obj.draw_projective_edges("lr1/deer.obj", 800, 800, 400, -400)
    obj.img.save("lr3/deer_lighted_colored_foved.jpg")
    obj.img.imshow()


def doLR4():
    obj = OBJ3DModel()
    obj.img.width = 1600
    obj.img.height = 1600
    obj.img.arr_init()
    obj.img.zBuffer_init()
    obj.draw_guro_edges("lr4/rabbit.obj")
    obj.img.save("lr4/rabbit_guro.jpg")
    obj.img.imshow()


if __name__ == "__main__":
    doLR4()
    print('done')
    pass
