


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import numpy as np
import matplotlib.pyplot as plt


#load data
X, y = load_iris(return_X_y=True)   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train decision tree
model_dt = DecisionTreeClassifier(
    criterion = 'gini',#entropy
    max_depth=3, #prevent overfitting 
    random_state=42
)

model_dt.fit(X_train, y_train)

#evalutate decision tree

y_pred = model_dt.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred): .2f}")

#Visualize decision tree
plt.figure(figsize=(10, 6))
plot_tree(model_dt, filled=True, feature_names=['SepalLength', 'SepaWdith', 'PetalLength', 'PetalWidth'])
plt.show()

#predict a sample

#sample1 = np.array([[5.9, 3, 5.1, 1.8]])
#pred = model_dt.predict(sample1)
#print(pred)