import numpy as np
from scipy.stats import entropy

def kl(w1, w2):
    return entropy(w1, w2, 2)

def js(w1, w2):
    r = (w1 + w2) / 2
    return 0.5*(entropy(w1, r, 2) + entropy(w2, r, 2))