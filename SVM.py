import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

numberImages = datasets.load_digits()

supportVectorClassifier = svm.SVC(gamma = 0.0001, C = 100) # Penalty parameter C of the error term.

x,y = numberImages.data[:-5], numberImages.target[:-5] # az utolso 10 elemet meghagyjuk tesztelesre ... ezert azokat kihagyjuk
supportVectorClassifier.fit(x,y)

predictedImage = numberImages.data[-4] # ez azt jelenti, hogy hatulrol a 6. elemet akarjuk megkapni

print "Assume the image is: ", supportVectorClassifier.predict(predictedImage)

plt.imshow(numberImages.images[-4],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()