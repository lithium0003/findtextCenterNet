import numpy as np
import re

modulo_list = [1091,1093,1097]
width = 1024
height = 1024
scale = 4
feature_dim = 100

def gaussian(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def sigmoid(x):
    return (np.tanh(x / 2) + 1) / 2

def softmax(x):
    mx = np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(x - mx)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator/denominator

def calcHist(im):
    agg = 1
    rHist, bins = np.histogram(im[...,0], 256 // agg, (0.,255.))
    gHist, bins = np.histogram(im[...,1], 256 // agg, (0.,255.))
    bHist, bins = np.histogram(im[...,2], 256 // agg, (0.,255.))

    maxPeakDiff = -1
    for hist in [rHist, gHist, bHist]:
        y = np.array(hist)
        x = np.linspace(0.,255.,len(y))

        if np.sum(y) == 0:
            continue

        idx = np.argsort(-y)
        mu_y = x[idx[0]]
        mean_y = np.sum(x * y) / np.sum(y)

        if mu_y > mean_y:
            peak1 = y[idx[0]:]
            x1 = x[idx[0]:]
            peak1 = np.concatenate([peak1[::-1],peak1[1:]], axis=0)
            x1 = np.concatenate([(2 * x1[0] - x1[::-1]),x1[1:]], axis=0)
        else:
            peak1 = y[:idx[0]+1]
            x1 = x[:idx[0]+1]
            peak1 = np.concatenate([peak1[:-1],peak1[::-1]], axis=0)
            x1 = np.concatenate([x1[:-1],(x1 + x1[-1])], axis=0)

        mu = np.sum(x1 * peak1) / np.sum(peak1)
        sigma = np.sqrt(np.sum((x1 - mu)**2 * peak1) / np.sum(peak1))
        fixmax = np.max(y[np.bitwise_and(mu + 10 > x, x > mu - 10)])

        neg_peak = gaussian(x, fixmax, mu, sigma + 10)
        fixy = y - neg_peak
        fixy[fixy < 0] = 0

        if np.sum(fixy) == 0:
            continue

        fix_diff = np.sum(np.abs(x - mu) * fixy) / np.sum(fixy)
        idx = np.argsort(-fixy)
        fix_maxx = np.abs(x[idx[0]] - mu)

        maxPeakDiff = max(maxPeakDiff, fix_diff, fix_maxx)

        if False:
            import matplotlib.pyplot as plt
            plt.subplot(2,1,1)
            plt.plot(x,y)
            plt.plot(x,gaussian(x, fixmax, mu, sigma + 10))
            plt.subplot(2,1,2)
            plt.plot(x,fixy)
            plt.vlines(mu, *plt.ylim(), 'r')
            plt.vlines(np.sum(x * fixy) / np.sum(fixy), *plt.ylim(), 'g')
            plt.show()

    return maxPeakDiff

def calc_predid(*args):
    m = modulo_list
    b = args
    assert(len(m) == len(b))
    t = []

    for k in range(len(m)):
        u = 0
        for j in range(k):
            w = t[j]
            for i in range(j):
                w *= m[i]
            u += w
        tk = (b[k] - u) % m[k]
        for j in range(k):
            tk *= pow(m[j], m[k]-2, m[k])
            #tk *= pow(m[j], -1, m[k])
        tk = tk % m[k]
        t.append(tk)
    x = 0
    for k in range(len(t)):
        w = t[k]
        for i in range(k):
            w *= m[i]
        x += w
    mk = 1
    for k in range(len(m)):
        mk *= m[k]
    x = x % mk
    return x

def decode_ruby(text):
    text = re.sub('\uFFF9(.*?)\uFFFA(.*?)\uFFFB',r'<ruby><rb>\1</rb><rp>(</rp><rt>\2</rt><rp>)</rp></ruby>', text)
    return text

def encode_rubyhtml(text):
    text = re.sub('<ruby><rb>(.*?)</rb><rp>\\(</rp><rt>(.*?)</rt><rp>\\)</rp></ruby>', '\uFFF9\\1\uFFFA\\2\uFFFB', text)
    return text
