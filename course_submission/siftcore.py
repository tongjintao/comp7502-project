import scipy
import numpy

#Provide utilities for SIFT

class siftcore(object):
    def __init__(self):
        self.sigma = 1.6

    def createdog(self,imagearr):
        re = [0,1,2,3]
        re[0] = self.diff(self.gs_blur(self.sigma, imagearr))
        for i in range(1,4):
            base = self.resize(re[i-1][2])
            re[i] = self.diff(self.gs_blur(self.sigma, base))
        return re

    def diff(self,images):
        diffArray = [0,1,2,3]
        for i in range(1,5):
            diffArray[i-1] = images[i] - images[i-1]
        return numpy.array(diffArray)

    def gs_blur(self,k,img):
        sig = [self.sigma,k*self.sigma,k*k*self.sigma,k*k*k*self.sigma,k*k*k*k*self.sigma]
        gsArray = [0,1,2,3,4]
        scaleImages = [0,1,2,3,4]
        
        for i in range(5):
            gsArray[i] = scipy.ndimage.filters.gaussian_filter(img,sig[i])
        return gsArray

    def resize(self,arr):
        H=0
        W=0
        if arr.shape[0]%2 == 0:
            H = arr.shape[0]/2
        else:
            H = 1+arr.shape[0]/2

        if arr.shape[1]%2 == 0:
            W = arr.shape[1]/2
        else:
            W = 1+arr.shape[1]/2
        
        new_arr = numpy.zeros((H,W),dtype = numpy.int)
        for i in range(H):
            for j in range(W):
                new_arr[i][j] = arr[2*i][2*j]
        return new_arr

    def dist(self,pair0,pair1,pair2):
        ax = pair0[0][0]
        ay = pair0[0][1]
        Ax = pair0[1][0]
        Ay = pair0[1][1]

        bx = pair1[0][0]
        by = pair1[0][1]
        Bx = pair1[1][0]
        By = pair1[1][1]

        cx = pair2[0][0]
        cy = pair2[0][1]
        Cx = pair2[1][0]
        Cy = pair2[1][1]
        
        dista_b = math.sqrt((ax-bx)**2+(ay-by)**2)
        dista_c = math.sqrt((ax-cx)**2+(ay-cy)**2)
        distb_c = math.sqrt((bx-cx)**2+(by-cy)**2)
        
        distA_B = math.sqrt((Ax-Bx)**2+(Ay-By)**2)
        distA_C = math.sqrt((Ax-Cx)**2+(Ay-Cy)**2)
        distB_C = math.sqrt((Bx-Cx)**2+(By-Cy)**2)
                            
        return dista_b,dista_c,distb_c,distA_B,distA_C,distB_C 


    def cal_area(self,x,y,z):
        p = (x+y+z)/2
        return math.sqrt(p*(p-x)*(p-y)*(p-z))