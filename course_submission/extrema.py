import numpy
import math
import scipy
import scipy.spatial

class extrema(object):

    def run(self,ims,pa):
    	arr = pa
        coordinates = []
        temp = {}
        H = [0,1,2,3]
        W = [0,1,2,3]

        for i in range(4):
            H[i] = len(ims[i][0])
            W[i] = len(ims[i][0][0])

        #Gaussian go through, then extract edges
        imx = numpy.zeros(arr.shape)
        imy = numpy.zeros(arr.shape)
        filters.gaussian_filter(arr, (3,3), (0,1), imx)
        filters.gaussian_filter(arr, (3,3), (1,0), imy)
        Wxx = filters.gaussian_filter(imx*imx, 3)
        Wxy = filters.gaussian_filter(imx*imy, 3)
        Wyy = filters.gaussian_filter(imy*imy, 3)
        Wdet = Wxx * Wyy - Wxy ** 2
        Wtr = Wxx + Wyy
        hess = []
        re = numpy.where(Wtr ** 2 / (Wdet + 0.00001) > 4.0)
        for i in range(len(re[0])):
            hess.append((re[0][i], re[1][i]))

        goodArea = [0,1,2]
        bad_points = list(set(hess) | set(pa))       #Filter out false key points
        bad = dict.fromkeys(bad_points, 0)

        #SIFT step 2, reference on OpenCV implementation
        for m in range(4):
            for n in range(1,3):
                for i in range(16,H[m]-16):
                    for j in range(16,W[m]-16):
                        if bad.has_key((i*2**m,j*2**m))==False :                               
                            goodArea[0] = ims[m][n-1][i-1:i+2,j-1:j+2]
                            goodArea[1] = ims[m][n][i-1:i+2,j-1:j+2]
                            goodArea[2] = ims[m][n+1][i-1:i+2,j-1:j+2]

                            maxLocal = numpy.array(goodArea).max()
                            minLocal = numpy.array(goodArea).min()

                            if (ims[m][n][i][j] == maxLocal) or (ims[m][n][i][j] == minLocal):
                                if temp.has_key((i*2**m,j*2**m)) == False:
                                    coordinates.append([int(i*2**m),int(j*2**m)])
                                    temp[(i*2**m,j*2**m)] = [i*2**m,j*2**m]
        return coordinates
