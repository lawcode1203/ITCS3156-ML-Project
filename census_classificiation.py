from ucimlrepo import fetch_ucirepo
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import numpy as np
import random

#Load the data
adult = fetch_ucirepo(id=2)

X = adult.data.features
y = adult.data.targets['income']

numerical_features = ['age',
                      'fnlwgt',
                      'education-num',
                      'capital-gain',
                      'capital-loss',
                      'hours-per-week'
                      ]

categorical_features = ['workclass',
                        'education',
                        'marital-status',
                        'occupation',
                        'relationship',
                        'race',
                        'sex',
                        'native-country'
                        ]


##### Preprocessing #####
oh = OneHotEncoder(sparse_output=False)
scaler = StandardScaler()
X_categorical_onehot = oh.fit_transform(X[categorical_features])
X_categorical_onehot = pd.DataFrame(X_categorical_onehot,
                                    columns=oh.get_feature_names_out())
X_numerical_scaled = scaler.fit_transform(X[numerical_features])
X_numerical_scaled = pd.DataFrame(X_numerical_scaled,
                                  columns=numerical_features)

X = pd.concat([X_categorical_onehot,X_numerical_scaled],axis=1)


def encode(x):
    if "<=50K" in x:
        return 0
    elif ">50K" in x:
        return 1

y = np.array(y.apply(encode))

##### Data Preparation #####

def train_test_validation_split(X,y, train_size, val_size, test_size):
    length = len(y)
    train_dist = int(train_size*length)
    test_dist = int(test_size*length)
    val_dist = int(val_size*length)
    
    X_shuffled, y_shuffled = shuffle(X, y, random_state=42)
    X_train, y_train = X[:train_dist], y[:train_dist]
    X_val, y_val = X[train_dist:train_dist+val_dist], y[train_dist:train_dist+val_dist]
    X_test, y_test = X[train_dist+val_dist:], y[train_dist+val_dist:]
    return X_train, X_val, X_test, y_train, y_val, y_test
    


# 50% for training, 20% for validation. 30% for testing
X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_split(X, y, 0.5, 0.2, 0.3)

##### Hyperparameter search - Logistic Regression #####

best_score =0.0
for c in [1.0, 0.5, 0.25, 0.125, 0.125/2, 0.125/4, 0.125/8]:
    for solver in ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']:
        log = LogisticRegression(solver=solver,random_state=42, C=c)
        log.fit(X_train, y_train)
        current_score = log.score(X_val, y_val)
        if current_score > best_score:
            print(c, solver, log.score(X_val, y_val))
            best_score = current_score


# Best hyperparameters found -> c = 0.03125, solver=liblinear

log = LogisticRegression(C=0.03125, solver='liblinear')

##### Hyperparameter search - Neural Network #####

best_score = 0.0

iterations = 30
for _ in range(iterations):
    depth = random.choice([1,2,3,4,5,6,7,8])
    activation = random.choice(['logistic','tanh','relu'])
    width = random.choice([8,16,32,64])
    alpha = random.choice([1e-3, 1e-4, 1e-5])
    batch_size = random.choice([16,32,64,128,256])
    lr = random.choice([1e-3,1e-4,1e-5])
    epochs = random.randint(2,200)
    mlp = MLPClassifier(hidden_layer_sizes = tuple([width]*depth), activation=activation,
                        solver='adam',
                        alpha=alpha,
                        batch_size=batch_size,
                        learning_rate_init=lr,
                        max_iter=epochs,
                        random_state=42)
    mlp.fit(X_train,y_train)
    current_score = mlp.score(X_val, y_val)
    if current_score > best_score:
        print(depth, activation, width, alpha, batch_size, lr, epochs, mlp.score(X_val, y_val))
        best_score = current_score

# Best model found -> depth: 4, activation: logistic, width: 8, alpha: 1e-4, batch_size:16
#                     lr:1e-3, epochs:72  
                        
mlp = MLPClassifier(hidden_layer_sizes=(8,8,8,8), activation='logistic', solver='adam',
                    alpha=1e-4, batch_size=16, learning_rate_init=1e-3, max_iter=72)

##### Model training #####
X_train_combined = np.concatenate((X_train,X_val))
y_train_combined =np.concatenate((y_train,y_val))

log.fit(X_train_combined, y_train_combined)
mlp.fit(X_train_combined, y_train_combined)


##### Model inference #####

y_hat_log = log.predict(X_test)
y_hat_mlp = mlp.predict(X_test)

log_accuracy = accuracy_score(y_test, y_hat_log)
mlp_accuracy = accuracy_score(y_test, y_hat_mlp)

print(f"Logistic Regression Model Test Accuracy: {log_accuracy}")
print(f"Multi-Layer Perceptron Model Test Accuracy: {mlp_accuracy}")

##### Confusion Matrix #####
# Code for the confusion matrix visualization was taken from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

disp_log = ConfusionMatrixDisplay.from_estimator(
    log,
    X_test,
    y_test,
    cmap=plt.cm.Blues,
    normalize="true")



plt.show()

disp_mlp = ConfusionMatrixDisplay.from_estimator(
    mlp,
    X_test,
    y_test,
    cmap=plt.cm.Blues,
    normalize="true")

plt.show()
