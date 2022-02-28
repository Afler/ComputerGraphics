from lr1.Coordinate import Coordinate


class Parser:
    @staticmethod
    def getVertexes(path):
        f = open(path, "r")
        list = f.read().split("\n")
        vertexList = []
        for row in list:
            if row.startswith("v "):
                vertexList.append(row)
        coordList = []
        for vertex in vertexList:
            coordinates = vertex.split(" ")
            coordList.append(Coordinate(float(coordinates[1]), float(coordinates[2]), float(coordinates[3])))
        return coordList

    @staticmethod
    def getPolygons(path):
        f = open(path, "r")
        list = f.read().split("\n")
        polygonList = []
        for row in list:
            if row.startswith("f "):
                polygonList.append(row)
        resultList = []
        for polRow in polygonList:
            vertexIndexes = polRow.split(" ")
            vertexIndex1 = vertexIndexes[1].split("/")[0]
            vertexIndex2 = vertexIndexes[2].split("/")[0]
            vertexIndex3 = vertexIndexes[3].split("/")[0]
            resultList.append([int(vertexIndex1), int(vertexIndex2), int(vertexIndex3)])
        return resultList
