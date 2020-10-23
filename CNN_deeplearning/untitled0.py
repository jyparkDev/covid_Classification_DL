import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

# import some data to play with
iris = datasets.load_iris()

X = iris.data
y = iris.target
class_names = iris.target_names
 
# Split the data into a training set and a test set

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)
print(classifier)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
y_dic = []
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Reds,
                                 normalize=normalize)
    print(disp)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    y_dic.append(disp.confusion_matrix)

plt.show()
y_dic = np.array(y_dic)
y_dic = y_dic[0]+y_dic[1]
y_dic1 = y_dic/2
print(y_dic)
print(y_dic1)

print(y_dic1.confusion_matrix)
