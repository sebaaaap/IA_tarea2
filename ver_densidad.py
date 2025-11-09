import pandas as pd

# Este script sirve para verificar la densidad de las clases en el dataset
df = pd.read_csv('dataset_t2.csv')
print(df['Rating_Category'].value_counts(normalize=True))
