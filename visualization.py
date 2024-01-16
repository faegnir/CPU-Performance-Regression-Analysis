#different plots adapted for MLR
"""
coefficients = model.coef_

#print the pie chart
sizes = abs(model.coef_.flatten())
labels = df.columns[1:-1]
colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'cyan', 'magenta', 'brown','gray'] 

plt.pie(sizes, labels=labels, colors=colors)
plt.title("Coefficients")
plt.show()

#print the bar graph
plt.bar(labels,sizes)
plt.ylabel('Coefficients')
plt.xticks(rotation=60,fontsize=10)
plt.show()


fitted_values = model.predict(X_train)
residuals = y_train - fitted_values
plt.scatter(fitted_values, residuals,s=10)
plt.axhline(y=2500, color='r', linestyle='-')
plt.axhline(y=-2500, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

plt.scatter(y_train, fitted_values,s=10)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


plt.plot(y_test,color = 'blue')

plt.plot(y_pred, color = 'orange')
plt.show()

residuals = y_train - fitted_values

sm.qqplot(residuals, line='s')
plt.show()
sns.pairplot(df)
plt.show()
"""

#RandomForestRegression Plots
"""
# Feature importance plot 
importances = rf_reg.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), df.columns[1:12][indices], rotation=90)
plt.title('Feature Importance - Random Forest Regression')
plt.show()

# scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred,s=14)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Regression')

plt.show()
"""


# NNR plots
"""
plt.scatter(y_test, y_pred_nn,s=14)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1.5)
plt.title('Neural Network Regression: Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

residuals = y_test - y_pred_nn
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

#The weights shown in the graph correspond to the 
#connections between the input and hidden layers (layer 0 to layer 1)
#connections between the hidden and output layers (layer 1 to layer 2).
coef = nn_reg.coefs_
for i in range(len(coef)):
    plt.hist(coef[i], bins=20)
    plt.title("Layer {} weight distribution".format(i+1))
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.show()

importance = nn_reg.coefs_[0]
feature_names = df.columns[1:12]
plt.barh(feature_names, importance[0])
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
"""

# Gradient Boosting Regression Plots
"""
plt.scatter(y_test, y_pred_gbr, s=14)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1.5)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Gradient Boosting Regression')
plt.show()

# Feature importances of Gradient Boosting Regression
feature_importance = gbr.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
features = df.columns[1:12]
plt.bar(features[sorted_idx], feature_importance[sorted_idx])
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Gradient Boosting Regression')
plt.xticks(rotation=90)
plt.show()

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(gbr, X_train, y_train.ravel(), cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label='Training Score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation Score')
plt.title('Gradient Boosting: Learning Curve')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend()
plt.show()
"""

# Plots SVR
"""
plt.scatter(y_pred_svr, y_pred_svr - y_test.ravel(), s=20)
plt.axhline(y=0, color='r', linestyle='-')
plt.hlines(y=0, xmin=0, xmax=50, lw=2)
plt.title('Residual Plot for SVR')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.show()

sns.distplot((y_test - y_pred_svr), color='blue')
plt.title('Distribution Plot')
plt.xlabel('Residuals')
plt.show()
"""