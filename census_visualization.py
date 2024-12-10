import seaborn as sea
from ucimlrepo import fetch_ucirepo
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA

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
def encode(x):
    if "<=50K" in x:
        return 0
    elif ">50K" in x:
        return 1

y = np.array(y.apply(encode))



# Show class imbalance
sea.histplot(y,bins=[0,0.5,1]);plt.show()
print(f"Proportion of >50K samples to <=50K samples: {sum(y)/len(y):.2f}")

# Show age distribution
sea.histplot(x=X['age']);plt.show()
print(f"Mean age: {sum(X['age'])/X.shape[0]:.2f}")


##### Scale the features and apply PCA #####
# PCA example taken from:
#https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py

scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X[numerical_features])
X_numerical_scaled = pd.DataFrame(X_numerical_scaled,
                                  columns=numerical_features)

pca = PCA(n_components=2)
pca_X = pca.fit_transform(X_numerical_scaled)
plt.scatter(x=pca_X[:,0],y=pca_X[:,1],c=y)
plt.show()

# Finally, do a seaborn pairplot of the numerical features
sea.pairplot(X_numerical_scaled);plt.show()
