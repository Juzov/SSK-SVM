import numpy as np
import math as math
from tail_recursion import tail_recursive, recurse
import sys
sys.setrecursionlimit(100000)

class StringSubsequenceKernel:
    def __init__(self, k, lambda_decay):
        self.k = k
        self.lambda_decay = lambda_decay

    def run_instance(self, s, t):
        self.k_dict = {}
        self.k_p_dict = {}
        self.k_pp_dict = {}
        return self.k_n(s, t, self.k)

    @tail_recursive
    def k_n(self, sx, t, i):
        if (sx, t, i) in self.k_dict:
            return self.k_dict[(sx, t, i)]
        elif (min(len(sx), len(t)) < i):
            self.k_dict[(sx, t, i)] = 0
            return 0

        s = sx[:-1]
        x = sx[-1]
        sumJ = 0

        for j in range(0, len(t)):
            if(t[j] == x):
                sumJ += self.k_p_n(s, t[:j], i - 1) * (self.lambda_decay ** 2)

        result = self.k_n(s, t, i) + sumJ
        self.k_dict[(sx, t, i)] = result
        return result

    @tail_recursive
    def k_p_n(self, sx, t, i):
        if (sx, t, i) in self.k_p_dict:
            return self.k_p_dict[(sx, t, i)]
        elif (i == 0):
            self.k_p_dict[(sx, t, i)] = 1
            return 1
        elif (min(len(sx), len(t)) < i):
            self.k_p_dict[(sx, t, i)] = 0
            return 0

        s = sx[:-1]
        x = sx[-1]

        result = self.lambda_decay * \
            self.k_p_n(s, t, i) + self.k_pp_n(sx, t, i)
        self.k_p_dict[(sx, t, i)] = result
        return result

    @tail_recursive
    def k_pp_n(self, sx, tz, i):
        if (sx, tz, i) in self.k_pp_dict:
            return self.k_pp_dict[(sx, tz, i)]

        elif (min(len(sx), len(tz)) < i):
            self.k_pp_dict[(sx, tz, i)] = 0
            return 0

        s = sx[:-1]
        x = sx[-1]

        t = tz[:-1]
        z = tz[-1]
        result = 0

        # same last elements
        if (x == z):
            result = self.lambda_decay * \
                (self.k_pp_n(sx, t, i) +
                 self.lambda_decay * self.k_p_n(s, t, i - 1))
        # different last elements
        elif (x != z):
            result = self.lambda_decay * self.k_pp_n(sx, t, i)

        self.k_pp_dict[(sx, tz, i)] = result
        return result


# string_s = 'science is organized knowledge'
# string_t = 'wisdom is organized life'
string_s = 'Subject nels fall conference dates next s nels conference jointly host harvard university mit hop set conference date conflict major nearby conference host conference next fall already set date please send email wednesday nov 16 martha jo mcginni mit'
string_t = 'Subject nels fall conference dates next s nels conference jointly host harvard university mit hop set conference date conflict major nearby conference host conference next fall already set date please send email wednesday nov 16 martha jo mcginni mit'

string_s = string_s.lower()
string_t = string_t.lower()

ssk_object = StringSubsequenceKernel(2, 0.5)
result = ssk_object.run_instance(string_s, string_t)