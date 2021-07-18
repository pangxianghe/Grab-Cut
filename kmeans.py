import numpy as np
import random
from matplotlib import pyplot as plt


class K_means:
    def __init__(self, mat, dim=3, k=2, max_iter=10):
        self.shape = list(mat.shape)
        self.rows = int(mat.size / dim)
        self.mat = mat.reshape(self.rows, dim)
        self.k = k
        self.types = np.zeros(self.rows, dtype=np.uint)
        self.max_iter = max_iter
        self.init_centers()

    def init_centers(self):
        center_index = []
        for i in range(self.k):
            while True:
                index = random.randint(0, self.rows - 1)
                if index not in center_index:
                    center_index.append(index)
                    break
        self.centers = self.mat[center_index]

    def determine_types(self):
        self.types = np.asarray(
            [np.asarray([((self.mat[i] - self.centers[j]) ** 2).sum() for j in range(self.k)]).argmin(0) for i in
             range(self.rows)])

    def refresh_centers(self):
        cluster_length = []
        for i in range(self.k):
            index = np.where(self.types == i)
            length = len(list(index)[0])
            cluster_length.append(length)
        cluster_length = np.asarray(cluster_length)

        for i in range(self.k):
            if cluster_length[i] == 0:
                # if a cluster empty, find the biggest cluster, find the farthest point from its center,
                # exclude it and form a new cluster.
                k = cluster_length.argmax(0)
                p = np.where(self.types == k)
                pixels = self.mat[p]
                index = np.asarray([((r - self.centers[k])**2).sum() for r in pixels]).argmax(0)
                index = list(p)[0][index]
                self.types[index] = i
                self.centers[i] = self.A[index]
            else:
                index = np.where(self.types == i)
                self.centers[i] = self.mat[index].sum(axis=0) / len(list(index)[0])

    def run(self):
        for i in range(self.max_iter):
            self.determine_types()
            self.refresh_centers()

    def output(self):
        data = []
        for i in range(self.k):
            index = np.where(self.types == i)
            data.append(self.mat[index])
        return data

    def plot(self):
        data_x = []
        data_y = []
        data_z = []
        for i in range(self.k):
            index = np.where(self.types == i)
            data_x.extend(self.mat[index][:, 0].tolist())
            data_y.extend(self.mat[index][:, 1].tolist())
            data_z.extend([i / self.k for j in range(len(list(index)[0]))])
        sc = plt.scatter(data_x, data_y, c=data_z, vmin=0, vmax=1, s=35, alpha=0.8)
        plt.colorbar(sc)
        plt.show()


if __name__ == '__main__':
    A = np.random.random([1000, 3, 2])
    kc = K_means(A, k=20, max_iter=10)
    kc.run()
    # kc.plot()
    r = kc.output()
    # print(r)
