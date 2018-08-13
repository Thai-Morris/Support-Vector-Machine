'''
This is a SVM aka suport vector machine learning algoritm
'''
#good for small datasets of less than a thousand units

#Donald Knuth said that premature optitization is the root of all evil, at least most of programming

# A hyperplace is a linear decision surface that splits the space into two pats; it is obvious that a hyperplane is a binary classifer.


__AUTHOR__ = '''Thai Morris'''

'''So I can do math'''
import numpy as np

'''to plot my data and odel visually'''

from matplotlib import pyplot as plt

'''step 1 - define my data

input data - of the form x value, Y value, Bias term'''

X = np.array([
        [-2, 4, -1],
        [4, 1, -1],
        [1, 6, -1],
        [2, 4, -1],
        [6, 2, -1],


])
'''
 associated output labels - first 2 examples are labeled '-1' and last 3 are...
'''
y = np.array([-1, -1, 1, 1, 1 ])

for d, sample in enumerate(X):
	#plot the negative samples (the first 2)
	if d < 2:
		plt.scatter(sample[0], sample[0], s=120, marker = '_', linewidths=2)
	#plot the postive samples (the last 3)
	else:
		plt.scatter(sample[0], sample[1], s=120, marker ='+', linewidths = 2)
		
	'''print a possible hyperplane, that is seperating thr wtwo classes I will add the two points and draw the line between them. I guess...'''
	
plt.plot([-2, 6], [6, 0.5])

def svm_sgd_plot(X, Y):
	
#intilize our svms weight vector with zeros (3 values)
	w = np.zeros(len(X[0]))
#the learning rate 
	eta = 1
#how many iterations to train for
	epochs = 100000
#store misclassifications so we can plot how they change over time
	errors = []

#traing part, graident descent part

	for epoch in range(1, epochs):
		error = 0
		for i, x in enumerate(X):
		#misclassification
			if (Y[i]*np.dot(X[i], w)) <1:
				w = w + eta * ( (X[i] * Y[i]) + (-2 *(1/epoch) *w))
			else:
			#correct classification, update our pipeline
			
				w = w + eta *(-2 * (1/epoch) * w)
		errors.append(error)
	
	#it's plotting time. The actual machine learning part
	
	plt.plot(errors, '|')
	plt.ylim(0.5, 1.5)
	plt.axes().set_yticklables([])
	plt.xlabel('Epoch')
	plt.ylabel('Misclasified')
	plt.show

	return w				
for d, sample in enumerate(X):
    # Plot the negative samples
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# Add my test samples
plt.scatter(2,2, s=120, marker='_', linewidths=2, color='yellow')
plt.scatter(4,3, s=120, marker='+', linewidths=2, color='blue')

# Print the hyperplane calculated by svm_sgd()
x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]

x2x3 =np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V,scale=1, color='blue')

w = svm_sgd_plot(X,y)
#they decrease over time! The SVM is learning the optimal hyperplane
			
			
			
			
			
