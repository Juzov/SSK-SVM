import numpy as np
import math as math
from tail_recursion import tail_recursive, recurse
import sys
sys.setrecursionlimit(100000)

class StringSubsequenceKernelWithPrune:
    def __init__(self, k, lambda_decay, theta):
        self.k = k
        self.lambda_decay = lambda_decay
        self.theta = theta

    def run_instance(self, s, t):
        self.k_dict = {}
        self.k_p_dict = {}
        self.k_pp_dict = {}
        return self.k_n(s, t, self.k, self.theta)

    @tail_recursive
    def k_n(self, sx, t, i, m):
        if (sx, t, i) in self.k_p_dict:
            return self.k_p_dict[(sx, t, i)]
        elif(min(len(sx), len(t)) < i):
            self.k_p_dict[(sx, t, i)] = 0
            return 0

        x = sx[-1]
        s = sx[:-1]
        sum_j = 0

        for j in range(0, len(t)):
            if(t[j] == x):
                sum_j += self.k_p_n(s, t[:j], i - 1,
                                    m - 2) * (self.lambda_decay ** 2)

        result = self.k_n(s, t, i, m) + sum_j
        self.k_p_dict[(sx, t, i)] = result
        return result

    @tail_recursive
    def k_p_n(self, sx, t, i, m):
        if (sx, t, i) in self.k_p_dict:
            return self.k_p_dict[(sx, t, i)]

        elif(i == 0):
            self.k_p_dict[(sx, t, i)] = 1
            return 1
        elif(min(len(sx), len(t)) < i):
            self.k_p_dict[(sx, t, i)] = 0
            return 0
        elif(m < (2 * i)):
            return 0

        x = sx[-1]
        s = sx[:-1]

        result = self.lambda_decay * \
            self.k_p_n(s, t, i, m - 1) + self.k_pp_n(sx, t, i, m)
        self.k_p_dict[(sx, t, i)] = result
        return result

    @tail_recursive
    def k_pp_n(self, sx, tz, i, m):
        if (sx, tz, i) in self.k_pp_dict:
            return self.k_pp_dict[(sx, tz, i)]

        if(min(len(sx), len(tz)) < i):
            self.k_pp_dict[(sx, tz, i)] = 0
            return 0

        x = sx[-1]
        s = sx[:-1]

        z = tz[-1]
        t = tz[:-1]
        sum_j = 0

        if(x == z):
            # same last elements
            sum_j += self.lambda_decay * \
                (self.k_pp_n(sx, t, i, m - 1) +
                 self.lambda_decay * self.k_p_n(s, t, i - 1, m - 2))
        elif(x != z):
            # different last elements
            count = 0
            for ti in range(len(t) - 1, -1, -1):
                removed_length = len(t) - ti
                if(t[ti] == x):
                    sum_j += (self.lambda_decay**(removed_length)) * \
                        self.k_pp_n(sx, t, i, m - removed_length)

        self.k_pp_dict[(sx, tz, i)] = sum_j
        return sum_j


# string_s = 'science is organized knowledge'
# string_t = 'wisdom is organized life'
string_s = 'Subject nels fall conference dates next s nels conference jointly host harvard university mit hop set conference date conflict major nearby conference host conference next fall already set date please send email wednesday nov 16 martha jo mcginni mit'
string_t = 'Subject nels fall conference dates next s nels conference jointly host harvard university mit hop set conference date conflict major nearby conference host conference next fall already set date please send email wednesday nov 16 martha jo mcginni mit'

string_s = string_s.lower()
string_t = string_t.lower()

ssk_object = StringSubsequenceKernelWithPrune(2, 0.5, 4)
result = ssk_object.run_instance(string_s, string_t)
