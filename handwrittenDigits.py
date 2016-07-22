import matplotlib.pyplot as plt
import numpy as np
import os

# Import of support vector machine (svm)
from sklearn import svm

"""--------------------------------SETTINGS---------------------------------"""

# Load data from .txt file
os.chdir("C:\\")
file = open("semeion.txt","r")
data = file.read()
file.close()

# Get index max (number of rows in the data)
def getIndexMax(data):
    dataSplitted = data.split("\n")
    return len(dataSplitted)


# Prepare data for fitting function
# Answer: if True, returns answers (labels)
# Training: if True, returns training samples (with no answers, only samples)
# Last: if True, returns only last training sample/answer. Useful for testing
def returnDataToUse(data,index,answers=False,training=False,last=False):
    dataSplitted = data.split("\n")

    # Check that the index is not bigger than our dataset
    if index > len(dataSplitted):
        print("Index out of bounds, index max:",len(dataSplitted))
        return 0

    # This bit of code returns answers
    if answers and not training:
        firstLine = dataSplitted[0][:-1].split(" ")
        firstAnsw = firstLine[256:]
        firstAnsw2 = [int(i) for i in firstAnsw]
        
        firstAnswInt = firstAnsw2.index(1)
        correctAnswers = [firstAnswInt]      

        i = 1
        while i < index:
            temp = dataSplitted[i][:-1].split(" ")
            temp2 = temp[256:]
            temp3 = [int(k) for k in temp2]
            temp4 = temp3.index(1)
            correctAnswers.append(temp4)
            i += 1

        completeAnswers = np.array(correctAnswers)

        if last:
            return completeAnswers[-1]
        else:
            return completeAnswers

    # This bit of code returns pure samples
    if training and not answers:
        firstLine = dataSplitted[0][:-1].split(" ")
        firstTraining = firstLine[:256]
        trainingArray = np.array([float(i) for i in firstTraining])

        i = 1
        while i < index:
            temp = dataSplitted[i][:-1].split(" ")
            temp2 = temp[:256]
            temp3 = np.array([float(k) for k in temp2])
            trainingArray = np.vstack((trainingArray,temp3))
            i += 1

        if last:
            return trainingArray[-1]
        else:
            return trainingArray

# This function displays the image of the number (sample at row x)
# and prints the answer the predictor should give us back
def displayImage(data,row):

    # Split each row
    dataSplitted = data.split("\n")

    # Get the 'rowth' row
    strings = dataSplitted[row]

    # Split row into numbers(string), and avoid blank at the end
    stringsSplitted = (strings[:-1]).split(" ")  

    # Get target and convert it into numbers, then in a numpy array
    risp = stringsSplitted[256:]
    rispInt = [int(i) for i in risp]
    rispNp = np.array(rispInt)

    # Print original data and number to guess in readable format
    print(rispInt)
    print("Number to guess:",rispInt.index(1),"\n")

    # Training array converted into float numbers
    training = stringsSplitted[:256]
    trainingFloat = [float(i) for i in training]

    # Building 16x16 (image) array
    #.
    i = 16
    k = 0
    img = np.array(trainingFloat[:16])
    while i <= len(trainingFloat):

        #print(i)
        #print(k)
        temp = np.array(trainingFloat[k:i])
        img = np.vstack((img,temp))
        
        k = i   
        i += 16

    # Plot image
    plt.imshow(img,cmap=plt.cm.gray_r,interpolation="nearest")
    plt.show()
        
        
"""------------------------------TRAINING------------------------------------"""
# FIX THE NUMBER OF TRAINING SAMPLES
trainingSamples = 1500

# Gamma: gradient descent parameter
clf = svm.SVC(gamma=0.01, C=100)

# Index max
print("Maximum index:",getIndexMax(data),"\n")

answerArray = returnDataToUse(data,trainingSamples,answers=True)
trainingAr = returnDataToUse(data,trainingSamples,training=True)

x,y = trainingAr,answerArray

#Fit the data
print("Training...")
clf.fit(x,y)


"""------------------------------Sample prediction--------------------------"""
# CHOOSE AN EXAMPLE TO PREDICT
example = 1555

predictQ = returnDataToUse(data,example,training=True,last=True)
predictA = returnDataToUse(data,example,answers=True,last=True)

print("Prediction:",clf.predict(predictQ))
print("Actual answer",predictA,"\n")

# Display the actual image
displayImage(data,example)


"""------------------------------Testing Function----------------------------"""
# Actual testing on residual samples (=samples not used for training)
print("Testing...")
correct = 0
wrong = 0
j = example+1
while j < getIndexMax(data):
    q = returnDataToUse(data,j,training=True,last=True)
    a = returnDataToUse(data,j,answers=True,last=True)
    p = clf.predict(q)
    if a == p[0]:
        correct += 1
    else:
        wrong += 1
    j += 1

print("Statistics, correct answers:",correct/(correct+wrong))