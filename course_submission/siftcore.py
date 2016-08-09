import scipy
import numpy

#Provide utilities for SIFT

class siftcore(object):

    def resize(self, arr):
        height = arr.shape[0]/2
        width = arr.shape[1]/2
        result = numpy.zeros((height, width), dtype = numpy.int)
        for i in range(height):
            for j in range(width):
                result[i][j] = arr[2*i][2*j]
        return result

    def createdog(self, img):
        result = []
        result[0] = self.diff(self.gaussianBlurInScale(img))
        for i in range(1,4):
            base = self.resize(result[i-1][2])
            result[i] = self.diff(self.gaussianBlurInScale(base))
        return result

    def diff(self, imgs):
        diffArray = [0,1,2,3]
        for i in range(1,5):
            diffArray[i-1] = imgs[i] - imgs[i-1]
        return numpy.array(diffArray)

    def gaussianBlurInScale(self, img):
        sigVal = 1.5
        #k = 2
        k = sigVal
        sig = [sigVal, k*sigVal, k*k*sigVal, k*k*k*sigVal, k*k*k*k*sigVal]
        result = []
        for i in range(5):
            result[i] = scipy.ndimage.filters.gaussian_filter(img,sig[i])
        return result

    def makeImgLowContrast(self, arr):
        height, width = arr.shape
        result = []
        for i in range(1, height-1):
            for j in range(1, width-1):
                temp = arr[i-1:i+2, j-1:j+2]
                stdVal = temp.std()
                meanVal = temp.mean()
                if stdVal == 0:
                    continue                
                elif abs((arr[i][j] - meanVal) / stdVal) > 0.33:
                    result.append((i,j))

        return result