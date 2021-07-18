import numpy as np
import cv2
from kmeans import K_means
from GCGraph import GCGraph


class GMM:
    def __init__(self, k=5):
        self.k = k
        self.weights = np.asarray([0. for _ in range(k)])
        self.means = np.asarray([[0., 0., 0.] for _ in range(k)])
        self.cov = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for _ in range(k)])
        self.cov_inv = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for _ in range(k)])
        self.cov_det = np.asarray([0. for _ in range(k)])
        # help to calculate pi
        self.pixel_count = np.asarray([0. for _ in range(k)])
        self.pixel_total_count = 0
        # help to calculate means
        self._sums = np.asarray([[0., 0., 0.] for _ in range(k)])
        # help to calculate cov
        self._prods = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for _ in range(k)])

    def _prob_pixel_component(self, pixel, ci):
        # probability of pixel belonging to the ci GMM
        inv = self.cov_inv[ci]
        det = self.cov_det[ci]
        t = pixel - self.means[ci]
        nt = np.asarray([t])
        mult = np.dot(inv, np.transpose(nt))
        mult = np.dot(nt, mult)
        return (1 / np.sqrt(det) * np.exp(-0.5 * mult))[0][0]

    def prob_pixel_GMM(self, pixel):
        # probability of pixel belonging to the GMM
        return sum([self._prob_pixel_component(pixel, ci) * self.weights[ci] for ci in range(self.k)])

    def most_likely_pixel_component(self, pixel):
        prob = np.asarray([self._prob_pixel_component(pixel, ci) for ci in range(self.k)])
        return prob.argmax(0)

    def add_pixel(self, pixel, ci):
        tp = pixel.copy().astype(np.float32)
        self._sums[ci] += tp
        tp.shape = (tp.size, 1)
        self._prods[ci] += np.dot(tp, np.transpose(tp))
        self.pixel_count[ci] += 1
        self.pixel_total_count += 1

    def learning(self):
        variance = 0.01
        for ci in range(self.k):
            n = self.pixel_count[ci]
            if n == 0:
                self.weights[ci] = 0
            else:
                self.weights[ci] = n / self.pixel_total_count
                self.means[ci] = self._sums[ci] / n
                nm = self.means[ci].copy()
                nm.shape = (nm.size, 1)
                self.cov[ci] = self._prods[ci] / n - np.dot(nm, np.transpose(nm))
                self.cov_det[ci] = np.linalg.det(self.cov[ci])
            while self.cov_det[ci] <= 0:
                # add the white noise to avoid singular covariance matrix
                # full rank(inv existence) guarantee
                self.cov[ci] += np.diag([variance for i in range(3)])
                self.cov_det[ci] = np.linalg.det(self.cov[ci])
            self.cov_inv[ci] = np.linalg.inv(self.cov[ci])


class GCClient:
    def __init__(self, img, k):
        self.k = k
        self.img = np.asarray(img, dtype=np.float32)
        self.img2 = img
        self.rows, self.cols = list(img.shape)[:2]

        self.gamma = 50
        self.lam = 9 * self.gamma
        self.beta = 0

        # used when interacting with users
        self._BLUE = [255, 0, 0]
        self._RED = [0, 0, 255]
        self._GREEN = [0, 255, 0]
        self._BLACK = [0, 0, 0]
        self._WHITE = [255, 255, 255]
        self._thickness = 3

        self._rectangle = False
        self._rect_over = False
        self._rect = [0, 0, 1, 1]  # coordinates chosen by users

        # default parameters
        self._GC_BGD = 0
        self._GC_FGD = 1
        self._GC_PR_BGD = 2
        self._GC_PR_FGD = 3

        self._mask = np.zeros([self.rows, self.cols], dtype=np.uint8)
        self._mask[:, :] = self._GC_BGD

        self.cal_beta()
        self.cal_nearby_weight()

    def cal_beta(self):
        # calculate beta from 8 neighbors,
        # used to adjust the difference of two nearby pixels in high or low contrast rate
        self._left_diff = self.img[:, 1:] - self.img[:, :-1]  # Left-difference
        self._upleft_diff = self.img[1:, 1:] - self.img[:-1, :-1]  # Up-Left difference
        self._up_diff = self.img[1:, :] - self.img[:-1, :]  # Up-difference
        self._upright_diff = self.img[1:, :-1] - self.img[:-1, 1:]  # Up-Right difference
        beta = (self._left_diff * self._left_diff).sum() + (self._upleft_diff * self._upleft_diff).sum() \
               + (self._up_diff * self._up_diff).sum() + (
                       self._upright_diff * self._upright_diff).sum()
        self.beta = 1 / (2 * beta / (4 * self.cols * self.rows - 3 * self.cols - 3 * self.rows + 2))

    def cal_nearby_weight(self):
        self.left_weight = np.zeros([self.rows, self.cols])
        self.upleft_weight = np.zeros([self.rows, self.cols])
        self.up_weight = np.zeros([self.rows, self.cols])
        self.upright_weight = np.zeros([self.rows, self.cols])
        for y in range(self.rows):
            for x in range(self.cols):
                color = self.img[y, x]
                if x >= 1:
                    diff = color - self.img[y, x-1]
                    # print(np.exp(-self.beta*(diff*diff).sum()))
                    self.left_weight[y, x] = self.gamma*np.exp(-self.beta*(diff*diff).sum())
                if x >= 1 and y >= 1:
                    diff = color - self.img[y-1, x-1]
                    self.upleft_weight[y, x] = self.gamma/np.sqrt(2) * np.exp(-self.beta*(diff*diff).sum())
                if y >= 1:
                    diff = color - self.img[y-1, x]
                    self.up_weight[y, x] = self.gamma*np.exp(-self.beta*(diff*diff).sum())
                if x+1 < self.cols and y >= 1:
                    diff = color - self.img[y-1, x+1]
                    self.upright_weight[y, x] = self.gamma/np.sqrt(2)*np.exp(-self.beta*(diff*diff).sum())

    def init_mask(self, event, x, y, flags, parm):
        if event == cv2.EVENT_RBUTTONDOWN:
            self._rectangle = True
            self._ix, self._iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._rectangle == True:
                self.img = self.img2.copy()
                cv2.rectangle(self.img,(self._ix,self._iy),(x,y),self._BLUE,2)
                self._rect = [min(self._ix,x),min(self._iy,y),abs(self._ix-x),abs(self._iy-y)]
                self.rect_or_mask = 0

        elif event == cv2.EVENT_RBUTTONUP:
            self._rectangle = False
            self._rect_over = True
            cv2.rectangle(self.img,(self._ix,self._iy),(x,y),self._BLUE,2)
            self._rect = [min(self._ix,x),min(self._iy,y),abs(self._ix-x),abs(self._iy-y)]
            self.rect_or_mask = 0
            self._mask[self._rect[1]+self._thickness:self._rect[1]+self._rect[3]-self._thickness, self._rect[0]+self._thickness:self._rect[0]+self._rect[2]-self._thickness] = self._GC_PR_FGD


    def init_with_kmeans(self):
        print(self.cols * self.rows)
        print(len(list(np.where(self._mask == 0))[1]))
        max_iter = 2
        self._bgd = np.where(np.logical_or(self._mask == self._GC_BGD, self._mask == self._GC_PR_BGD))
        self._fgd = np.where(np.logical_or(self._mask == self._GC_FGD, self._mask == self._GC_PR_FGD))
        self.BGD_pixels = self.img[self._bgd]
        self.FGD_pixels = self.img[self._fgd]
        KMB = K_means(self.BGD_pixels, dim=3, k=self.k, max_iter=max_iter)
        KMF = K_means(self.FGD_pixels, dim=3, k=self.k, max_iter=max_iter)
        KMB.run()
        KMF.run()
        self._BGD_by_components = KMB.output()
        self._FGD_by_components = KMF.output()

        self.BGD_GMM = GMM()
        self.FGD_GMM = GMM()
        for ci in range(self.k):
            for pixel in self._BGD_by_components[ci]:
                self.BGD_GMM.add_pixel(pixel, ci)
            for pixel in self._FGD_by_components[ci]:
                self.FGD_GMM.add_pixel(pixel, ci)
        self.BGD_GMM.learning()
        self.FGD_GMM.learning()

    def assign_GMM_components(self):
        self.components_index = np.zeros([self.rows, self.cols], dtype=np.uint)
        for y in range(self.rows):
            for x in range(self.cols):
                pixel = self.img[y, x]
                if self._mask[y, x] == self._GC_BGD or self._mask[y, x] == self._GC_PR_BGD:
                    self.components_index[y, x] = self.BGD_GMM.most_likely_pixel_component(pixel)
                else:
                    self.components_index[y, x] = self.FGD_GMM.most_likely_pixel_component(pixel)

    def learn_GMM_parameters(self):
        for ci in range(self.k):
            bgd_ci = np.where(np.logical_and(self.components_index == ci,
                                             np.logical_or(self._mask == self._GC_BGD, self._mask == self._GC_PR_BGD)))
            fgd_ci = np.where(np.logical_and(self.components_index == ci,
                                             np.logical_or(self._mask == self._GC_FGD, self._mask == self._GC_PR_FGD)))
            for pixel in self.img[bgd_ci]:
                self.BGD_GMM.add_pixel(pixel, ci)
            for pixel in self.img[fgd_ci]:
                self.FGD_GMM.add_pixel(pixel, ci)
        self.BGD_GMM.learning()
        self.FGD_GMM.learning()

    def construct_graph(self, lam):
        vertex_count = self.cols * self.rows
        edge_count = 2 * (4 * vertex_count - 3 * (self.rows + self.cols) + 2)
        self.graph = GCGraph(vertex_count, edge_count)
        for y in range(self.rows):
            for x in range(self.cols):
                vertex_index = self.graph.add_vertex()
                color = self.img[y, x]
                if self._mask[y, x] == self._GC_PR_BGD or self._mask[y, x] == self._GC_PR_FGD:
                    fromSource = -np.log(self.BGD_GMM.prob_pixel_GMM(color))
                    toSink = -np.log(self.FGD_GMM.prob_pixel_GMM(color))
                elif self._mask[y, x] == self._GC_BGD:
                    fromSource = 0
                    toSink = lam
                else:
                    fromSource = lam
                    toSink = 0
                self.graph.add_term_weights(vertex_index, fromSource, toSink)

                if x > 0:
                    w = self.left_weight[y, x]
                    self.graph.add_edges(vertex_index, vertex_index - 1, w, w)
                if x > 0 and y > 0:
                    w = self.upleft_weight[y, x]
                    self.graph.add_edges(vertex_index, vertex_index - self.cols - 1, w, w)
                if y > 0:
                    w = self.up_weight[y, x]
                    self.graph.add_edges(vertex_index, vertex_index - self.cols, w, w)
                if x < self.cols - 1 and y > 0:
                    w = self.upright_weight[y, x]
                    self.graph.add_edges(vertex_index, vertex_index - self.cols + 1, w, w)

    def estimate_segmentation(self):
        a = self.graph.max_flow()
        for y in range(self.rows):
            for x in range(self.cols):
                if self._mask[y, x] == self._GC_PR_BGD or self._mask[y, x] == self._GC_PR_FGD:
                    if self.graph.insource_segment(y * self.cols + x):  # Vertex Index
                        self._mask[y, x] = self._GC_PR_FGD
                    else:
                        self._mask[y, x] = self._GC_PR_BGD

    def iter(self, n):
        for _ in range(n):
            self.assign_GMM_components()
            self.learn_GMM_parameters()
            self.construct_graph(self.lam)
            self.estimate_segmentation()

    def run(self):
        self.init_with_kmeans()
        self.iter(1)
