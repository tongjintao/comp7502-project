import numpy
import math

#Gen descriptor based on calculated feature values

class descriptor(object):
	def __init__(self):
		self.eps = 0.000034

	def creatDes(self, features, imarr):
            desDict = {}
	    arr = imarr.astype(numpy.float64)
	    desNum = len(features)
		
	    for i in range(desNum):
        	desDict[(features[i][0],features[i][1])] = self.allocate(features[i][0],features[i][1],arr)

	    return desDict


	def direction(self,i,j,imarr):
		mij = math.sqrt((imarr[i+1,j]-imarr[i-1,j])**2 
				+(imarr[i,j+1]-imarr[i,j-1])**2)
		theta = math.atan((imarr[i,j+1]-imarr[i,j-1])
				/(imarr[i+1,j]-imarr[i-1,j]+self.eps))

		return mij,theta


	def allocate(self,i,j,imarr):
		vec = [0]*16
		vec[0] = self.localdir(i-8,j-8,imarr)
		vec[1] = self.localdir(i-8,j,imarr)
		vec[2] = self.localdir(i-8,j+8,imarr)
		vec[3] = self.localdir(i-8,j+16,imarr)

		vec[4] = self.localdir(i,j-8,imarr)
		vec[5] = self.localdir(i,j,imarr)
		vec[6] = self.localdir(i,j+8,imarr)
		vec[7] = self.localdir(i,j+16,imarr)

		vec[8] = self.localdir(i+8,j-8,imarr)
		vec[9] = self.localdir(i+8,j,imarr)
		vec[10] = self.localdir(i+8,j+8,imarr)
		vec[11] = self.localdir(i+8,j+16,imarr)

		vec[12] = self.localdir(i+16,j-8,imarr)
		vec[13] = self.localdir(i+16,j,imarr)
		vec[14] = self.localdir(i+16,j+8,imarr)
		vec[15] = self.localdir(i+16,j+16,imarr)

		return [val for subl in vec for val in subl]

	def localdir(self,i,j,imarr):
		P = math.pi
		localDir = [0]*18

		for b in range(i-8,i):
			for c in range(j-8,j):
				m,t = self.direction(b,c,imarr)
				if t>=P*-9/18 and t<=P*-8/18:
					localDir[0]+=m
				if t>P*-8/18 and t<=P*-7/18:
					localDir[1]+=m
				if t>P*-7/18 and t<=P*-6/18: 
					localDir[2]+=m
				if t>P*-6/18 and t<=P*-5/18:
					localDir[3]+=m	
				if t>P*-5/18 and t<=P*-4/18:
					localDir[4]+=m
				if t>P*-4/18 and t<=P*-3/18:
					localDir[5]+=m
				if t>P*-3/18 and t<=P*-2/18:
					localDir[6]+=m	
				if t>P*-2/18 and t<=P*-1/18:
					localDir[7]+=m
				if t>P*-1/18 and t<=0:
					localDir[8]+=m
				if t>0 and t<=P*1/18: 
					localDir[9]+=m
				if t>P*1/18 and t<=P*2/18:
					localDir[10]+=m	
				if t>P*2/18 and t<=P*3/18:
					localDir[11]+=m
				if t>P*3/18 and t<=P*4/18:
					localDir[12]+=m
				if t>P*4/18 and t<=P*5/18:
					localDir[13]+=m	
				if t>P*5/18 and t<=P*6/18:
					localDir[14]+=m
				if t>P*6/18 and t<=P*7/18:
					localDir[15]+=m
				if t>P*7/18 and t<=P*8/18:
					localDir[16]+=m
				if t>P*8/18 and t<=P*9/18:
					localDir[17]+=m

		return localDir
		