from flask import Flask, render_template, request
import joblib
import yfinance as yf
import numpy as np

app = Flask(__name__)

# Loading the model at the top so it's ready when the app starts
try:
    model = joblib.load('model.pkl')
except:
    print("Error: model.pkl not found. Make sure it is in the same folder as app.py")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    input_value = None
    error = None

    if request.method == 'POST':
        # 1. Getting the ticker from the user (e.g., AAPL or GC=F)
        input_value = request.form['stock_input'].upper()

        try:
            # 2. Fetching the latest day of data
            stock_data = yf.Ticker(input_value)
            df = stock_data.history(period="1d")

            if df.empty:
                error = f"No data found for symbol '{input_value}'. Check the ticker name."
            else:
                # 3. Extracting the 4 features of Ridge model was trained on
                # Features in the notebook: Open, High, Low, Volume
                features = np.array([[
                    df['Open'].iloc[-1],
                    df['High'].iloc[-1],
                    df['Low'].iloc[-1],
                    df['Volume'].iloc[-1]
                ]])

                # 4. Using the model to predict the Close price
                pred_raw = model.predict(features)
                prediction = round(float(pred_raw[0]), 2)

        except Exception as e:
            error = f"An error occurred: {str(e)}"

    return render_template('index.html', 
                           prediction=prediction, 
                           input_value=input_value, 
                           error=error)

if __name__ == "__main__":
    app.run(debug=True)