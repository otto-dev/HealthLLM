import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df_500 = pd.read_csv('/Users/chongzhang/PycharmProjects/ai_for_health_final/classification/result2000.csv')
df_y = pd.read_csv('/Users/chongzhang/PycharmProjects/ai_for_health_final/label and feature/label_data.csv',
                   encoding='utf-8')
df_y = df_y.iloc[0:2000]
df = df_500.copy()

df_X = df.reset_index(drop=True)
df_y = df_y.reset_index(drop=True)

data = pd.concat([df_X, df_y], axis=1)
rows_with_all_zeros = df[(df == 0).all(axis=1)]
drop_index = rows_with_all_zeros.index
data = data.drop(index=drop_index)

list_q = []
dict_c = {}
for i in range(0, 40):
    y = data.iloc[:, -42 + i:-41 + i]
    y = y.iloc[:, -1]

    rows_with_0_or_1 = y[y.isin([0, 1])]
    count_0 = (rows_with_0_or_1 == 0).sum()
    count_1 = (rows_with_0_or_1 == 1).sum()
    k = 1 - count_1 / (count_0 + count_1)
    if k > 0.95:
        print('hi')
    else:
        X = data.iloc[:, 0:55]
        y = data.iloc[:, -42 + i:-41 + i]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict(X_test)
        print(k)
        print("Accuracy Score:", accuracy_score(y_test, y_pred))
        dict_c[k] = "Accuracy Score:" + str(accuracy_score(y_test, y_pred))

        k = accuracy_score(y_test, y_pred) - 1 + count_1 / (count_0 + count_1)
        list_q.append(k)

sum = 0
for i in list_q:
    sum += i
print("Accuracy Score:", sum / len(list_q))

print(dict_c)
'''
for i in range(0,40):
    X = data.iloc[:, 0:56]
    y = data.iloc[:, -42+i:-41+i]
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)
    num = accuracy_score(y_test, y_pred)
    list_q.append(num)

    # print("Accuracy Score:", accuracy_score(y_test, y_pred))
filtered_values = [value for value in list_q if value != 1.0]

sum = 0
for i in filtered_values:
    sum+=i
print("Accuracy Score:", sum/ len(filtered_values))
'''
