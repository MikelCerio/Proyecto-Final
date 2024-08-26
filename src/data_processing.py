import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Definir la ruta relativa para cargar el archivo CSV original
input_file_path = os.path.join('..', 'data', 'raw', 'online_shoppers_intention.csv')

# Cargar los datos
clientes = pd.read_csv(input_file_path)

# Preprocesamiento de datos
meses_dict = {"Feb": 1, "Mar": 2, "May": 3, "June": 4, "Jul": 5, "Aug": 6, "Sep": 7, "Oct": 8, "Nov": 9, "Dec": 10}
clientes["Mes_Num"] = clientes["Month"].map(meses_dict)
clientes.drop(columns=['Month'], inplace=True)
clientes['Weekend'] = clientes['Weekend'].astype(int)
clientes['Revenue'] = clientes['Revenue'].astype(int)
clientes_new = pd.get_dummies(clientes, columns=['VisitorType'], dtype=int)

# Seleccionar características relevantes
features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 
            'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
X = clientes_new[features]
y = clientes_new['Revenue']

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir las rutas relativas para guardar los archivos CSV de train y test
train_file_path = os.path.join('..', 'data', 'processed', 'train.csv')
test_file_path = os.path.join('..', 'data', 'processed', 'test.csv')

# Crear el directorio 'processed' si no existe
os.makedirs(os.path.dirname(train_file_path), exist_ok=True)

# Guardar los conjuntos de datos en archivos CSV
train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

train_data.to_csv(train_file_path, index=False)
test_data.to_csv(test_file_path, index=False)

print(f"Archivos de train y test guardados en:\n{train_file_path}\n{test_file_path}")
