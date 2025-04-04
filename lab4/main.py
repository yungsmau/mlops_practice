from catboost.datasets import titanic
import pandas as pd

version = int(input('Выберите версию датасета, которую хотите сгенирировать - 1, 2, или 3:\n'))

# Исходный датасет
df, _ = titanic()

if version == 1:
    # Версия 1, комммит 1
    df = df[['Pclass', 'Sex', 'Age']]
    df.to_csv('titanic_csa.csv', index=False)

elif version == 2:
    # Версия 2, коммит 2
    df = df[['Pclass', 'Sex', 'Age']]
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df.to_csv('titanic_csa.csv', index=False)

elif version == 3:
    # Версия 3, коммит 3
    df = df[['Pclass', 'Sex', 'Age']]
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df = pd.get_dummies(df, columns=['Sex'])
    df.to_csv('titanic_csa.csv', index=False)