import data_lib as dl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import numpy as np

df = dl.readSparkDf("Cleaned/Sales_cleaned.csv")
print(df.head())
print(df.count())
print(df.isnull().sum())

#regression gonna predict sales amount 
saleColoumns_df = df[['Quantity', 'Category', 'Discount', 'Region','Segment' , 'Ship Mode', 'State','Sub-Category']]
sales_label = df['Sales']

preprocessor = ColumnTransformer(
    transformers = [
        
       ('onehot', OneHotEncoder(sparse =False, handle_unknown='ignore'), ['Segment','Sub-Category','Category', 'Region', 'State']),

       ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['Ship Mode'])

    ],
    remainder='passthrough'
)

# scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    saleColoumns_df,
    sales_label,
    test_size=0.2,
    random_state=42
)

model = Pipeline(
    steps = 
    [
        ('preprocessor', preprocessor),
        # ('scaler', scaler),
        ('model',
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42
            )
            
            # GradientBoostingRegressor(
            #     n_estimators=200, 
            #     learning_rate=0.1,
            #     max_depth=3, 
            #     random_state=42)
        )
    ]
)

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

print(f"Train R2: {r2_score(y_train, y_pred_train):.3f}")
print(f"Test  R2: {r2_score(y_test,  y_pred_test):.3f}")
print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):.2f}")
print(f"Test  MAE: {mean_absolute_error(y_test,  y_pred_test):.2f}")

cv_scores = cross_val_score(model, saleColoumns_df, sales_label, cv=5, scoring='r2')
print(f"CV R2 mean: {np.mean(cv_scores):.3f}  std: {np.std(cv_scores):.3f}")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")