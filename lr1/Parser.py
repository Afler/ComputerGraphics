from lr1.Coordinate import Coordinate


class Parser:
    @staticmethod
    def getVertexes(path):
        f = open(path, "r")
        list = f.read().split("\n")
        vertexList = []
        vertexListVN = []
        for row in list:
            if row.startswith("v "):
                vertexList.append(row)
        coordList = []
        for vertex in vertexList:
            coordinates = vertex.split(" ")
            coordList.append(Coordinate(float(coordinates[1]), float(coordinates[2]), float(coordinates[3])))
        for row in list:
            if row.startswith("vn "):
                vertexListVN.append(row)
        coordListVN = []
        for vertex in vertexListVN:
            coordinates = vertex.split(" ")
            coordListVN.append(Coordinate(float(coordinates[1]), float(coordinates[2]), float(coordinates[3])))
        return coordList, coordListVN

    # @staticmethod
    # def getNormals(path):
    #     f = open(path, "r")
    #     list = f.read().split("\n")
    #     vertexListVN = []
    #     for row in list:
    #         if row.startswith("vn "):
    #             vertexListVN.append(row)
    #     coordListVN = []
    #     for vertex in vertexList:
    #         coordinates = vertex.split(" ")
    #         coordListVN.append(Coordinate(float(coordinates[1]), float(coordinates[2]), float(coordinates[3])))
    #     return coordListVN

    @staticmethod
    def getPolygons(path):
        f = open(path, "r")
        list = f.read().split("\n")
        polygonList = []
        for row in list:
            if row.startswith("f "):
                polygonList.append(row)
        resultList = []
        normalList = []
        for polRow in polygonList:
            vertexIndexes = polRow.split(" ")
            vertexIndex1 = vertexIndexes[1].split("/")[0]
            vertexIndex2 = vertexIndexes[2].split("/")[0]
            vertexIndex3 = vertexIndexes[3].split("/")[0]
            normalIndex1 = vertexIndexes[1].split("/")[2]
            normalIndex2 = vertexIndexes[2].split("/")[2]
            normalIndex3 = vertexIndexes[3].split("/")[2]
            resultList.append([int(vertexIndex1), int(vertexIndex2), int(vertexIndex3)])
            normalList.append([int(normalIndex1), int(normalIndex2), int(normalIndex3)])
        return resultList, normalList
