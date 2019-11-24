import numpy as np
from sys import argv
from typing import List, Tuple
from matplotlib import pyplot as plt
from matplotlib import animation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    inputFilename: str = argv[1]
    outputFilename: str = argv[2]
    if not inputFilename or not outputFilename:
        print('You forgot the INPUT and OUTPUT files!')
        exit()

    A: List[float] = []
    B: List[float] = []
    label: List[int] = []
    colors: List[str] = []
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    def plot_decision_boundary(predictor):
        num = 100
        x = np.linspace(0, 4, num)
        y = np.linspace(0, 4, num)
        X = []
        Y = []
        C = []
        X.extend(A)
        Y.extend(B)
        C.extend(colors)

        for i in range(num):
            for j in range(num):
                X.append(x[i])
                Y.append(y[j])
                C.append('#0a8af215' if predictor.predict([[x[i], y[j]]])[0] == 1 else '#f20e0a15')

        plt.scatter(X, Y, c=C)
        plt.show()

    # outputFile = open(outputFilename, "w")

    with open(inputFilename, "r") as inputFile:
        for line in inputFile:
            (A_i, B_i, label_i) = tuple(line.split(','))
            if A_i == 'A':
                continue
            (A_i, B_i) = (float(A_i), float(B_i))
            label_i = int(label_i)
            A.append(A_i)
            B.append(B_i)
            label.append(label_i)
            colors.append('#0a8af2' if label_i == 1 else '#f20e0a')
    

    # Split data
    X = np.column_stack((A, B))
    [X_train, X_test, label_train, label_test] = train_test_split(X, label, train_size=0.6, stratify=label)

    # SVM with linear kernel
    # svc_linear = SVC(kernel='linear')
    # gs_svc_linear = GridSearchCV(svc_linear, { 'C': [0.1, 0.5, 1, 5, 10, 50, 100] }, cv=5)
    # gs_svc_linear.fit(X_train, label_train)
    # plot_decision_boundary(gs_svc_linear)
    # outputFile.write(','.join(('svm_linear', str(gs_svc_linear.best_score_), str(gs_svc_linear.score(X_test, label_test))))+'\n')

    # SVM with polynomial kernel
    # svc_poly = SVC(kernel='poly')
    # gs_svc_poly = GridSearchCV(svc_poly, { 'C': [0.1, 1, 3], 'degree': [4, 5, 6], 'gamma': [0.1, 0.5] }, cv=5)
    # gs_svc_poly.fit(X_train, label_train)
    # plot_decision_boundary(gs_svc_poly)
    # outputFile.write(','.join(('svm_polynomial', str(gs_svc_poly.best_score_), str(gs_svc_poly.score(X_test, label_test))))+'\n')

    # SVM with RBF kernel
    # svc_rbf = SVC(kernel='rbf')
    # gs_svc_rbf = GridSearchCV(svc_rbf, { 'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10] }, cv=5)
    # gs_svc_rbf.fit(X_train, label_train)
    # plot_decision_boundary(gs_svc_rbf)
    # outputFile.write(','.join(('svm_rbf', str(gs_svc_rbf.best_score_), str(gs_svc_rbf.score(X_test, label_test))))+'\n')

    # Logistic regression
    # lr = LogisticRegression()
    # gs_lr = GridSearchCV(lr, { 'C': [0.1, 0.5, 1, 5, 10, 50, 100] }, cv=5)
    # gs_lr.fit(X_train, label_train)
    # plot_decision_boundary(gs_lr)
    # outputFile.write(','.join(('logistic', str(gs_lr.best_score_), str(gs_lr.score(X_test, label_test))))+'\n')
    
    # K-Nearest Neighbors
    # knn = KNeighborsClassifier()
    # gs_knn = GridSearchCV(knn, { 'n_neighbors': list(range(1, 51)), 'leaf_size': list(range(5, 61, 5)) }, cv=5)
    # gs_knn.fit(X_train, label_train)
    # plot_decision_boundary(gs_knn)
    # outputFile.write(','.join(('knn', str(gs_knn.best_score_), str(gs_knn.score(X_test, label_test))))+'\n')

    # Decision Trees
    dt = DecisionTreeClassifier()
    gs_dt = GridSearchCV(dt, { 'max_depth': list(range(1, 51)), 'min_samples_split': list(range(2, 11)) }, cv=5)
    gs_dt.fit(X_train, label_train)
    plot_decision_boundary(gs_dt)
    # outputFile.write(','.join(('decision_tree', str(gs_dt.best_score_), str(gs_dt.score(X_test, label_test))))+'\n')

    # Random Forest
    # rf = RandomForestClassifier()
    # gs_rf = GridSearchCV(rf, { 'max_depth': list(range(1, 51)), 'min_samples_split': list(range(2, 11)) }, cv=5)
    # gs_rf.fit(X_train, label_train)
    # plot_decision_boundary(gs_rf)
    # outputFile.write(','.join(('random_forest', str(gs_rf.best_score_), str(gs_rf.score(X_test, label_test))))+'\n')

    # outputFile.close()

    
