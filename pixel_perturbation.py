import os
from pickle import TRUE
from tkinter.tix import Tree
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import random
import matplotlib.patches as mpatches
from hyperlpr_py3 import pipline as pp
from hyperlpr_py3 import api as api

StrLUT = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
          "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
          "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
          "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
          "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
          "W": 61, "X": 62, "Y": 63, "Z": 64, "港": 65, "学": 66, "O": 67, "使": 68, "警": 69, "澳": 70, "挂": 71}

CharSet = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
           "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
           "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
           "Q", "R", "S", "T", "U", "V", "W", "X",
           "Y", "Z", "港", "学", "O", "使", "警", "澳", "挂"]
StrLUT = {v: k for k, v in StrLUT.items()}

# Possible colors for the output charts or figures.
colors = ["b", "g", "r", "c", "m", "y", "k"]
num_change = []  # Record the change in the number of individuals between different iterations
rgb_change = []  # Record the change value of RGB for each iteration

AutomaticStop = TRUE


def eachFile(filepath):
    lista = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        lista.append(child)
    return lista


def substract(a_list, b_list):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        new_list.append(a_list[i]-b_list[i])
    return new_list


def add(a_list, b_list):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        new_list.append(a_list[i]+b_list[i])
    return new_list


def multiply(a, b_list):
    b = len(b_list)
    new_list = []
    for i in range(0, b):
        new_list.append(a * b_list[i])
    return new_list


def comp(elem):
    return elem[1]


class DiffEvolution:
    def __init__(self, np=200, f=0.5, cr=0.7, g=100, lenx=5, rgb_up=200, rgb_down=50, iter_max=6):
        self.NP = np                # number of population
        self.F = f                  # scaling factor
        self.CR = cr                # crossing rate
        # the number of individual properties with default value of 5, which are (x,y) and (r,g,b)
        self.NUM_PROPERTY = lenx
        # the maximum value of RGB at initialization.
        self.RGB_MAX = rgb_up
        # the minimum value of RGB at initialization.
        self.RGB_MIN = rgb_down

        self.generation = g         # number of generation
        self.iter_max = iter_max  # the number of iteration
        # the flag for HyperLPR classification. 0 stands for chinese license characters, 1 stands for numbers, 2 stands for letters.
        self.flag = 3
        self.ori_res = ''           # the original classification result.
        self.target_res = ''        # the targeted classification result.
        # Temporary classification result generated during the evaluation.
        self.temp_res = ''
        # Confidence of temporary classification result.
        self.temp_conf = 0.0

    def loadImg(self, path):
        self.image = cv2.imread(path)
        self.image = cv2.resize(self.image, (18, 35))
        self.maxh = self.image.shape[0]
        self.maxw = self.image.shape[1]

    def object_function(self, x):
        res = np.zeros((self.maxh, self.maxw, 3), dtype=np.uint8)
        for i in range(0, 3):
            res[x[4], x[3], i] = x[i] - self.image[x[4], x[3], i]
        res = cv2.add(res, self.image)
        name, confidence, offset, res_set = pp.segmentation.singleDect(
            res, self.flag)
        correct_conf = res_set[CharSet.index(self.ori_res)-offset]
        if len(self.target_res) != 0:
            target_conf = res_set[CharSet.index(self.target_res)-offset]
        else:
            sort_set = np.sort(res_set, axis=0)
            target_conf = sort_set[len(sort_set)-2]
            loc = np.where(res_set == target_conf)[0][0] + offset
            self.target_res = StrLUT[loc]
        # print(name, confidence)
        return correct_conf, target_conf

    def evaluation(self, x):
        res = np.zeros((self.maxh, self.maxw, 3), dtype=np.uint8)
        for j in range(0, len(x)):
            for i in range(0, 3):
                res[x[j][4], x[j][3], i] = x[j][i] - \
                    self.image[x[j][4], x[j][3], i]
        res = cv2.add(res, self.image)
        cv2.imwrite('./images_out/Perturbed_Image.png', res)
        res_set = pp.segmentation.singleDect(res, self.flag)
        self.temp_res = res_set[0]
        self.temp_conf = res_set[1]
        print(self.temp_res, self.temp_conf)
        return self.temp_res, self.temp_conf

    # clamp RGB value
    def clampRGB(self, np):
        for j in range(0, len(np)):
            if j == 3:
                if np[j] > self.maxw-1:
                    np[j] = self.maxw-1
                if np[j] < 0:
                    np[j] = 0
            elif j == 4:
                if np[j] > self.maxh-1:
                    np[j] = self.maxh-1
                if np[j] < 0:
                    np[j] = 0
            elif j != 3 and j != 4 and np[j] > 255:
                np[j] = 255
            elif j != 3 and j != 4 and np[j] < 0:
                np[j] = 0
        np = [int(k) for k in np]
        return np

    # initialtion
    def initialtion(self, NP, max_h, max_w):
        np_list = []   # population
        for i in range(0, NP):
            x_list = []   # individual
            for j in range(0, self.NUM_PROPERTY):
                if j == 3:
                    x_list.append(int(random.random() * max_w))
                elif j == 4:
                    x_list.append(int(random.random() * max_h))
                else:
                    x_list.append(int(self.RGB_MIN + random.random()
                                  * (self.RGB_MAX - self.RGB_MIN)))
            np_list.append(x_list)
        return np_list

    # mutation
    def mutation(self, np_list):
        v_list = []
        for i in range(0, len(np_list)):
            r1 = random.randint(0, len(np_list)-1)
            while r1 == i:
                r1 = random.randint(0, len(np_list)-1)
            r2 = random.randint(0, len(np_list)-1)
            while r2 == r1 | r2 == i:
                r2 = random.randint(0, len(np_list)-1)
            r3 = random.randint(0, len(np_list)-1)
            while r3 == r2 | r3 == r1 | r3 == i:
                r3 = random.randint(0, len(np_list)-1)

            v_list.append(add(np_list[r1], multiply(
                self.F, substract(np_list[r2], np_list[r3]))))
            v_list[i] = self.clampRGB(v_list[i])
        return v_list

    # crossover
    def crossover(self, np_list, v_list):
        u_list = []
        for i in range(0, len(np_list)):
            vv_list = []
            for j in range(0, self.NUM_PROPERTY):
                if (random.random() <= self.CR) | (j == random.randint(0, self.NUM_PROPERTY - 1)):
                    vv_list.append(v_list[i][j])
                else:
                    vv_list.append(np_list[i][j])
            u_list.append(vv_list)
            u_list[i] = self.clampRGB(u_list[i])
        return u_list

    # selection
    def selection(self, u_list, np_list):
        changeScale = []
        for i in range(0, len(np_list)):
            origin = self.object_function(np_list[i])
            changed = self.object_function(u_list[i])
            if changed[0] < origin[0] and changed[1] > origin[1]:
                np_list[i] = u_list[i]
            else:
                np_list[i] = np_list[i]
        return np_list

    def updateCurPos(self, path):
        self.flag = 3
        pass

    def updateOriRes(self):
        res_set = pp.segmentation.singleDect(self.image, self.flag)
        self.ori_res = res_set[0]

    def setTargetRes(self, str):
        self.target_res = str

    def saveImg(self, path, x):
        res = np.zeros((self.maxh, self.maxw, 3), dtype=np.uint8)
        for j in range(0, len(x)):
            for i in range(0, 3):
                res[x[j][4], x[j][3], i] = x[j][i]
        res = cv2.add(res, self.image)
        cv2.imwrite(path, res)

    def getNum(self, np_list):
        pos = [(i[3], i[4]) for i in np_list]
        sin_pos = set(pos)
        return len(sin_pos)

    def getAverageRGB(self, np_list):
        rgb = [(i[0], i[1], i[2]) for i in np_list]
        r = 0
        g = 0
        b = 0
        for i in rgb:
            r += i[0]
            b += i[1]
            g += i[2]

        r /= len(rgb)
        g /= len(rgb)
        b /= len(rgb)

        return [r, g, b]


def plotRGB(index, de):
    global rgb_change
    legends = []
    x_label = np.arange(0, de.generation, 1)

    if len(rgb_change) <= de.generation:
        lastElement = rgb_change[len(rgb_change)-1]
        for i in range(len(rgb_change), de.generation):
            rgb_change.append(lastElement)

    patch_r = mpatches.Patch(color='r', label="Red Channel")
    patch_b = mpatches.Patch(color='b', label="Blue Channel")
    patch_g = mpatches.Patch(color='g', label="Green Channel")
    plt.plot(x_label, [i[0] for i in rgb_change], color='r')
    plt.plot(x_label, [i[1] for i in rgb_change], color='b')
    plt.plot(x_label, [i[2] for i in rgb_change], color='g')

    legends = [patch_r, patch_b, patch_g]
    plt.legend(handles=legends, loc='best')
    plt.xlabel('generation')
    plt.ylabel('the average RGB value of population')
    plt.savefig('./images_out/Process/rgb'+str(index+1)+'.png')
    # plt.show()
    pass


def plotConverge(de: DiffEvolution):
    global num_change
    legends = []
    x_label = np.arange(0, de.generation, 1)

    for index in range(0, de.iter_max):
        patch = mpatches.Patch(
            color=colors[index], label="iteration "+str(index+1))
        plt.plot(x_label, np.array(num_change)[
            de.generation*index:de.generation*(index+1)], color=colors[index])
        legends.append(patch)

    plt.legend(handles=legends, loc='best')
    plt.xlabel('generation')
    plt.ylabel('the number of individual')
    plt.savefig('./images_out/Process/converge.png')
    # plt.show()
    pass


def attack(img_path, target):
    isAttackEnd = False
    de = DiffEvolution(iter_max=6)
    de.loadImg(img_path)
    de.updateOriRes()
    de.setTargetRes(target)
    np_list = de.initialtion(de.NP, de.maxh, de.maxw)
    global rgb_change
    global num_change

    for j in range(de.iter_max):
        tmp_numChange = 0
        rgb_change = []
        for i in range(0, de.generation):
            print("Iteration", j+1, " Generation",
                  i+1, " Target \"" + target + "\"")
            v_list = de.mutation(np_list)
            u_list = de.crossover(np_list, v_list)
            np_list = de.selection(u_list, np_list)
            name, confidence = de.evaluation(np_list)
            if AutomaticStop == TRUE:
                x = set([i[3] for i in np_list])
                y = set([i[4] for i in np_list])

                if len(x) == 1 and len(y) == 1:
                    break
                if len(x) <= 4 and len(y) <= 4 and de.temp_res == de.target_res and de.temp_conf > 0.7:
                    isAttackEnd = TRUE
                    break
            tmp_numChange += 1
            num_change.append(de.getNum(np_list))
            rgb_change.append(de.getAverageRGB(np_list))

        if tmp_numChange < de.generation:
            lastElement = num_change[len(num_change)-1]
            for i in range(tmp_numChange, de.generation):
                num_change.append(lastElement)

        if isAttackEnd == TRUE:
            break
        if j != de.iter_max-1:
            de.loadImg("images_out/Perturbed_Image.png")
            np_list = de.initialtion(de.NP, de.maxh, de.maxw)

        plotRGB(j, de)
        plt.cla()

    # plotConverge(de)
    # plt.cla()

    de.saveImg("images_out/Perturbed_Example.png", np_list)
    final_name, final_confidence = de.evaluation(np_list)
    return [final_name, final_confidence]


if __name__ == '__main__':
    attack("images_in/testsets/1.png", 'F')

    # cv2.imshow('image', b)
    # cv2.waitKey(0)
