import numpy
import math

#Gen descriptor based on calculated feature values

class descriptor(object):

	def creatDes(self, features, arr):
		result = {}
		for i in range(len(features)):
			result[(features[i][0],features[i][1])] = self.allocFeat(features[i][0],features[i][1],arr)
		return result

	def direction(self,i,j,arr):
		v1 = math.sqrt((arr[i+1,j]-arr[i-1,j])**2 +(arr[i,j+1]-arr[i,j-1])**2)
		v2 = math.atan((arr[i,j+1]-arr[i,j-1])/(arr[i+1,j]-arr[i-1,j] + 0.000034))
		return v1,v2

	#Array of features of descriptors
	def allocFeat(self,i,j,arr):
		result = []
		result[0] = self.localFeat(i-8,j-8,arr)
		result[1] = self.localFeat(i-8,j,arr)
		result[2] = self.localFeat(i-8,j+8,arr)
		result[3] = self.localFeat(i-8,j+16,arr)
		result[4] = self.localFeat(i,j-8,arr)
		result[5] = self.localFeat(i,j,arr)
		result[6] = self.localFeat(i,j+8,arr)
		result[7] = self.localFeat(i,j+16,arr)
		result[8] = self.localFeat(i+8,j-8,arr)
		result[9] = self.localFeat(i+8,j,arr)
		result[10] = self.localFeat(i+8,j+8,arr)
		result[11] = self.localFeat(i+8,j+16,arr)
		result[12] = self.localFeat(i+16,j-8,arr)
		result[13] = self.localFeat(i+16,j,arr)
		result[14] = self.localFeat(i+16,j+8,arr)
		result[15] = self.localFeat(i+16,j+16,arr)
		return result

	#Det feature calc
	def localFeat(self,i,j,arr):
		result = []
		for b in range(i-8,i):
			for c in range(j-8,j):
				m,t = self.direction(b,c,arr)
				if t>=math.pi*-9/18 and t<=math.pi*-8/18:
					result[0]+=m
				if t>math.pi*-8/18 and t<=math.pi*-7/18:
					result[1]+=m
				if t>math.pi*-7/18 and t<=math.pi*-6/18: 
					result[2]+=m
				if t>math.pi*-6/18 and t<=math.pi*-5/18:
					result[3]+=m	
				if t>math.pi*-5/18 and t<=math.pi*-4/18:
					result[4]+=m
				if t>math.pi*-4/18 and t<=math.pi*-3/18:
					result[5]+=m
				if t>math.pi*-3/18 and t<=math.pi*-2/18:
					result[6]+=m	
				if t>math.pi*-2/18 and t<=math.pi*-1/18:
					result[7]+=m
				if t>math.pi*-1/18 and t<=0:
					result[8]+=m
				if t>0 and t<=math.pi*1/18: 
					result[9]+=m
				if t>math.pi*1/18 and t<=math.pi*2/18:
					result[10]+=m	
				if t>math.pi*2/18 and t<=math.pi*3/18:
					result[11]+=m
				if t>math.pi*3/18 and t<=math.pi*4/18:
					result[12]+=m
				if t>math.pi*4/18 and t<=math.pi*5/18:
					result[13]+=m	
				if t>math.pi*5/18 and t<=math.pi*6/18:
					result[14]+=m
				if t>math.pi*6/18 and t<=math.pi*7/18:
					result[15]+=m
				if t>math.pi*7/18 and t<=math.pi*8/18:
					result[16]+=m
				if t>math.pi*8/18 and t<=math.pi*9/18:
					result[17]+=m

		return result
		