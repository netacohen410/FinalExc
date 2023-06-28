# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 00:41:53 2023

@author: user
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import madlan_data_prep
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)


path = "C:\\Users\\user\\Desktop\\נטע\\לימודים\\שנה ג\\סמסטר ב\\ניתוח נתונים מתקדם\\"
filename = "Train_Set.xlsx"
data = path + filename
df = pd.read_excel(data)

#שימוש בפונקציה לטיפול בנתונים יש להכניס את שם של הדאטה
data = madlan_data_prep.prepare_data(df)

#חלוקת הנתונים למשתנים בלתי תלויים ומשתנה תלוי
X = data.drop(columns=["price"])
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#הגדרה של עמודות מספריות וקטגוריות
num_cols = [col for col in X_train.columns if X_train[col].dtypes!='O']
cat_cols = [col for col in X_train.columns if (X_train[col].dtypes=='O')]

#שימוש בפיפליין עבור הערכים המספריים והקטגוריים
numerical_pipeline = Pipeline([
    ('numerical_imputation', SimpleImputer(strategy='median', add_indicator=False)),
    ('scaling', StandardScaler())])

categorical_pipeline = Pipeline([
    ('categorical_imputation', SimpleImputer(strategy='constant', add_indicator=False, fill_value='missing')),
    ('one_hot_encoding', OneHotEncoder(sparse=False, handle_unknown='ignore'))])

column_transformer = ColumnTransformer([
     ('numerical_preprocessing', numerical_pipeline, num_cols),
    ('categorical_preprocessing', categorical_pipeline, cat_cols)], remainder='drop')

pipe_preprocessing_model = Pipeline([
    ('preprocessing_step', column_transformer),
    ('model', ElasticNet(alpha=0.01))])


# שימוש ב10-Fold Cross Validation והערכת מדדי ביצוע
cv_scores = cross_val_score(pipe_preprocessing_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)
mae_scores = -cross_val_score(pipe_preprocessing_model, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')

print("CV RMSE:", np.mean(rmse_scores), "(", np.std(rmse_scores), ")")
print("CV MAE:", np.mean(mae_scores), "(", np.std(mae_scores), ")")

# אימון המודל על Train Data
pipe_preprocessing_model.fit(X_train, y_train)

# ביצוע תחזית על Test Data
y_pred = pipe_preprocessing_model.predict(X_test)

# חישוב מדדי הביצוע
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)

print("Test RMSE:", test_rmse)
print("Test MAE:", test_mae)


import pickle
pickle.dump(pipe_preprocessing_model, open("trained_model.pkl","wb"))
