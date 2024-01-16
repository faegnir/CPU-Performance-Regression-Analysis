#algorithm to find best train-test split
"""
best_r2 = 0
loop = 500
tsize = [0.2,0.25,0.3,0.35,0.4]
print(tsize)

#finding optimum train-test ratio
for a in tsize:
    r2 = 0
    for i in range(loop):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=a)
        sc_X = MinMaxScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)

        sc_y = MinMaxScaler()
        y_train = sc_y.fit_transform(y_train) 
        y_test = sc_y.fit_transform(y_test)
        
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
            
        sc_y = StandardScaler()
        y_train = sc_y.fit_transform(y_train)
        y_test = sc_y.transform(y_test)
        
    
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)  

        r2 += r2_score(y_test, y_pred)

    print(r2/loop) 
    r2 /= loop

    if r2 > best_r2:
        best_ratio = a
        best_r2 = r2

print("\nBest train-test ratio:", best_ratio)
print("Best R2 score:", best_r2)
"""

