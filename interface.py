import gradio as gr
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# upload your pre-trained model
model_path = "C:\Users\balki\Downloads\model.pkl"  # verify your path
with open(model_path, 'rb') as file:
    model = joblib.load(file)

# Preprocessing functions
def preprocessing_func(input_data):
    input_data['amount'] = input_data['amount'].astype('str')
    input_data['amount'] = input_data['amount'].str.replace('-', '')
    input_data['amount'] = input_data['amount'].str.replace('$', '')
    input_data['amount'] = pd.to_numeric(input_data['amount'])
    
    for col in ['use_chip', 'card_type']:
        input_data[col] = input_data[col].astype(str).str.strip().str.replace('\t', '', regex=True)
    
    cat_columns = input_data.select_dtypes(include=['object', 'category']).columns
    data_fraud1 = pd.get_dummies(input_data, columns=cat_columns, drop_first=False)
    
    
    scaler = StandardScaler()
    numerical_features = ['amount', 'current_age', 'credit_score', 'longitude', 'latitude']
    data_fraud1[numerical_features] = scaler.fit_transform(data_fraud1[numerical_features])
    
    return data_fraud1

# Function to detect fraud transactions
def detect_fraud(file):
    # upload an excel file containing the data of your transaction
    df = pd.read_excel(file.name)
    
    # Apply the preprocessing
    df_preprocessed = preprocessing_func(df)
    
    # Prediction
    predictions = model.predict(df_preprocessed)
    
    # Add predictions to the dataframe
    df['Num transaction'] = [f"Transaction {i + 1}" for i in range(len(predictions))]
    df['Etat'] = ["Frauduleuse" if pred == 1 else "Pas frauduleuse" for pred in predictions]
    df['Couleur'] = ["🔴" if pred == 1 else "🟢" for pred in predictions]  # Utilisation de symboles pour indiquer les couleurs



    # Return important columns
    return df[["Num transaction","Etat", "Couleur"]]  # Remplacez "Transaction ID" par la colonne existante dans votre dataset

# Gradio dashboard
iface = gr.Interface(
    fn=detect_fraud,
    inputs=gr.File(label="Téléchargez le fichier Excel des transactions"),  # Entrée : fichier .xlsx
    outputs=gr.DataFrame(label="Résultats des Transactions"),  # Sortie sous forme de DataFrame
    title="Modèle de détection de Transaction Frauduleuse",
    description=(
        "Ce modèle prédit si une transaction est frauduleuse ou non en se basant sur un modèle de classification binaire Random Forest."
        "Les transactions frauduleuses sont marquées par un cercle rouge (🔴), et les autres par un cercle vert (🟢)."
    )
)

# Launch dashboard
if __name__ == "__main__":
    iface.launch()
