import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import tkinter as tk
from tkinter import messagebox

# Data Preprocessing function
def load_and_clean_data():
    data = pd.read_csv('bengaluru_house_prices.csv')
    
    # Dropping unnecessary columns
    data = data.drop(columns=['area_type', 'availability', 'society', 'balcony'])
    
    # Filling null values
    data['size'] = data['size'].fillna('2 BHK')
    data['bath'] = data['bath'].fillna(data['bath'].median())
    
    # Extracting BHK from the 'size' column
    data['bhk'] = data['size'].str.split().str.get(0).astype(int)
    
    # Converting the 'total_sqft' column to a numerical value
    def convertRange(x):
        temp = x.split('-')
        if len(temp) == 2:
            return (float(temp[0]) + float(temp[1])) / 2
        try:
            return float(x)
        except:
            return None

    data['total_sqft'] = data['total_sqft'].apply(convertRange)
    
    # Removing rows where total_sqft per bhk is less than 300
    data = data[((data['total_sqft'] / data['bhk']) >= 300)]
    
    # Removing outliers
    data['price_per_sqft'] = data['price'] * 100000 / data['total_sqft']
    
    def remove_outliers_sqft(df):
        df_output = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            gen_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
            df_output = pd.concat([df_output, gen_df], ignore_index=True)
        return df_output

    data = remove_outliers_sqft(data)
    
    # Dropping 'price_per_sqft' as we no longer need it
    data = data.drop(columns=['price_per_sqft', 'size'])
    
    # Simplifying 'location' column by grouping rare locations
    data['location'] = data['location'].apply(lambda x: x.strip())
    location_count = data['location'].value_counts()
    location_count_less_10 = location_count[location_count <= 10]
    data['location'] = data['location'].apply(lambda x: 'Other' if x in location_count_less_10 else x)

    return data

# Model training function
def train_model():
    data = load_and_clean_data()
    
    # Defining the input features (X) and the target (y)
    X = data.drop(columns=['price'])
    y = data['price']
    
    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Creating a pipeline with OneHotEncoder and StandardScaler
    column_trans = make_column_transformer((OneHotEncoder(sparse_output=False), ['location']),
                                           remainder='passthrough')
    
    # Setting `with_mean=False` in StandardScaler to handle sparse matrices
    scaler = StandardScaler(with_mean=False)
    
    # Choosing Lasso Regression (can be changed to LinearRegression() or Ridge())
    model = Lasso()
    
    # Creating the pipeline
    pipe = make_pipeline(column_trans, scaler, model)
    
    # Fitting the model
    pipe.fit(X_train, y_train)
    
    # Testing the model
    y_pred = pipe.predict(X_test)
    print(f"R2 Score: {r2_score(y_test, y_pred)}")
    
    return pipe

# GUI Application using Tkinter
def predict_price():
    try:
        location = location_var.get()
        total_sqft = float(sqft_var.get()) 
        bhk = int(bhk_var.get())
        bath = int(bath_var.get())

        input_data = pd.DataFrame([[location, total_sqft, bhk, bath]], 
                                  columns=['location', 'total_sqft', 'bhk', 'bath'])
        
        prediction = model_pipeline.predict(input_data)[0]
        messagebox.showinfo("Prediction", f"Estimated Price: {prediction:.2f} lakhs")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Setting up the Tkinter window
root = tk.Tk()
root.title("House Price Predictor")

# Defining variables
location_var = tk.StringVar()
sqft_var = tk.StringVar()
bhk_var = tk.StringVar()
bath_var = tk.StringVar()

# Creating labels and entry fields
tk.Label(root, text="Location").grid(row=0, column=0)
tk.Entry(root, textvariable=location_var).grid(row=0, column=1)

tk.Label(root, text="Total Square Feet").grid(row=1, column=0)
tk.Entry(root, textvariable=sqft_var).grid(row=1, column=1)

tk.Label(root, text="BHK").grid(row=2, column=0)
tk.Entry(root, textvariable=bhk_var).grid(row=2, column=1)

tk.Label(root, text="Bath").grid(row=3, column=0)
tk.Entry(root, textvariable=bath_var).grid(row=3, column=1)

# Creating a button to trigger the prediction
tk.Button(root, text="Predict Price", command=predict_price).grid(row=4, columnspan=2)

# Training the model before launching the GUI
model_pipeline = train_model()

# Running the Tkinter event loop
root.mainloop()
