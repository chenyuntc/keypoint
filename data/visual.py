# encoding:utf-8
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from config import opt
from numpy import ma


def visualize_heat_maps(heat_maps, img):
    for i in range(15):
        print i
        visualize_heat_map(heat_maps[:, :, i], img)


def visualize_heat_map(heatmap, img):
    stride = opt.stride
    heatmap = cv.resize(
        heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv.INTER_CUBIC)
    # visualization
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(img[:, :, [2, 1, 0]])
    axarr[1].imshow(heatmap, alpha=.5)
    axarr[2].imshow(img[:, :, [2, 1, 0]])
    axarr[2].imshow(heatmap, alpha=.5)
    plt.show()


def visualize_pafs_single_figure(pafs, img):

    stride = opt.stride
    idx = 0
    plt.figure()
    for i in range(13):
        U = pafs[:, :, idx]
        V = pafs[:, :, idx + 1]
        U = cv.resize(
            U, (0, 0), fx=stride, fy=stride, interpolation=cv.INTER_CUBIC)
        V = cv.resize(
            V, (0, 0), fx=stride, fy=stride, interpolation=cv.INTER_CUBIC)
        U = U * -1
        X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
        M = np.zeros(U.shape, dtype='bool')
        M[U**2 + V**2 < 0.5 * 0.5] = True
        U = ma.masked_array(U, mask=M)
        V = ma.masked_array(V, mask=M)

        # 1

        plt.imshow(img[:, :, [2, 1, 0]], alpha=.5)
        s = 5
        Q = plt.quiver(
            X[::s, ::s],
            Y[::s, ::s],
            U[::s, ::s],
            V[::s, ::s],
            scale=50,
            headaxislength=4,
            alpha=.5,
            width=0.001,
            color='r')
        idx += 2
    plt.show()


def visualize_pafs(pafs, img):
    stride = opt.stride
    idx = 0
    assert (pafs.shape[2]%2==0)
    for i in range(pafs.shape[2]/2):
        print (i+1)
        U = pafs[:, :, idx]
        V = pafs[:, :, idx + 1]
        U = cv.resize(
            U, (0, 0), fx=stride, fy=stride, interpolation=cv.INTER_CUBIC)
        V = cv.resize(
            V, (0, 0), fx=stride, fy=stride, interpolation=cv.INTER_CUBIC)
        visualize_paf(U, V, img)
        idx += 2


def visualize_paf(U, V, img):

    U = U * -1
    # V = vector_f_y[:, :, 17]
    X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
    M = np.zeros(U.shape, dtype='bool')
    M[U**2 + V**2 < 0.5 * 0.5] = True
    U = ma.masked_array(U, mask=M)
    V = ma.masked_array(V, mask=M)

    # 1
    plt.figure()
    plt.imshow(img[:, :, [2, 1, 0]], alpha=.5)
    s = 5
    Q = plt.quiver(
        X[::s, ::s],
        Y[::s, ::s],
        U[::s, ::s],
        V[::s, ::s],
        scale=70,
        headaxislength=4,
        alpha=.5,
        width=0.001,
        color='r')

    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    plt.show()