import numpy
import math
import scipy
import scipy.spatial
import siftcore
import descriptor
import extrema

class sift(object):
    def __init__(self):
        self.distanceThresh = 0.000002

    def getDescriptor(self, img):
    	img = numpy.array(img)
    	core = siftcore.siftcore()
    	des = descriptor.descriptor()
    	ex = extrema.extrema()
    	imgDog = core.creatdog(img)
    	feat = ex.run(imgDog, img)
    	return des.creatDes(feat, img)

    def match(self,p,s):
        pDes = getDescriptor(p)
        sDes = getDescriptor(s)
        tree = scipy.spatial.cKDTree(sDes.values())
        slocList = sDes.keys()
        pDict = {}
        sDict = {}
        result = {}

        for p in pDes.keys():
            x = pDes[p]
            re = tree.query(x,k=2,eps=self.distanceThresh,p=2,
                distance_upper_bound=numpy.inf)

            if re[0][1]==0 or (re[0][1]!=0 and re[0][0]/re[0][1] < 0.88):
                pLoc = p
                sLoc = slocList[re[1][0]]
                distance = re[0][0]
                
                if sDict.has_key(sLoc)==False:
                    result[(pLoc,sLoc)] = distance
                    pDict[pLoc] = sLoc
                    sDict[sLoc] = pLoc
                
                elif distance < result.get((sDict[sLoc],sLoc)):
                    del result[(sDict[sLoc],sLoc)]
                    result[(pLoc,sLoc)] = distance
                    del pDict[sDict[sLoc]]
                    pDict[pLoc] = sLoc
                    sDict[sLoc] = pLoc

        return sorted(result.items(), reverse=False, key=lambda d: d[1])
