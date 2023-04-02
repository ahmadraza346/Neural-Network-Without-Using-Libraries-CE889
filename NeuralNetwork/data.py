import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from random import shuffle

# reading the data from the game
data = pd.read_csv('game_data.csv', names=['X1', 'X2', 'Y1', 'Y2'])
# removing all the null values
nan_value = float("NaN")
data.replace("", nan_value)
data.dropna()
# removing duplicates
data.drop_duplicates(keep='first')
# shuffling data
# shuffle(data)
# scaling the data between 0 and 1 using minmaxscalar
scalar = MinMaxScaler(feature_range=(0, 1))
scaled_data = scalar.fit_transform(data)
data2 = pd.DataFrame(scaled_data, columns=data.columns)
# data2 = data
# splitting X and Y data
x = data2.drop(columns=['Y1', 'Y2'])
y = data2.drop(columns=['X1', 'X2'])
# splitting data between test training,testing and validation
x_train, x_validate_test, y_train, y_validate_test = train_test_split(x, y, train_size=0.7)
x_validate, x_test, y_validate, y_test = train_test_split(x_validate_test, y_validate_test, test_size=0.5)
# writing data to csv files
x_train.to_csv('x_train.csv', index=False)

y_train.to_csv('y_train.csv', index=False)

x_validate.to_csv('x_validate.csv', index=False)
y_validate.to_csv('y_validate.csv', index=False)
x_test.to_csv('x_test.csv',index=False)
y_test.to_csv('y_test.csv',index=False)
