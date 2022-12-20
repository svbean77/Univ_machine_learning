#기계학습응용 2주차 과제
import numpy as np
from random import randint
#1번
num1 = np.array([[randint(0, 10) for _ in range(5)] for _ in range(4)])

#2번
num2 = np.transpose(num1)

#3번
num3 = num1 ** 2

#4번
num4 = num1 + num3

#5번
num5 = num2 @ num1
