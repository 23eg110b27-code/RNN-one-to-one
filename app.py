import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam


# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="RNN One-to-One Stock Prediction",
    layout="centered"
)

st.title("📈 RNN One-to-One Stock Price Prediction")


# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("stock_market_trends_5000.csv")
    return df


df = load_data()

st.subheader("Dataset Preview")

# Avoid pyarrow issue
st.text(df.head().to_string())


# -------------------------------
# Data Preparation
# -------------------------------

features = [
    "open_price",
    "high_price",
    "low_price",
    "volume"
]

target = "close_price"


X = df[features]
y = df[[target]]


# Scaling
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()


X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)


# RNN Input Shape
# (samples, timestep, features)

X_rnn = X_scaled.reshape(
    X_scaled.shape[0],
    1,
    X_scaled.shape[1]
)


X_train, X_test, y_train, y_test = train_test_split(
    X_rnn,
    y_scaled,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# Build Model Function
# -------------------------------

@st.cache_resource
def create_model():

    model = Sequential()

    model.add(
        SimpleRNN(
            50,
            activation="tanh",
            input_shape=(1,4)
        )
    )

    model.add(Dense(1))


    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
    )

    return model


model = create_model()


# -------------------------------
# Train Model
# -------------------------------

if "trained" not in st.session_state:
    st.session_state.trained = False


if st.button("Train RNN Model"):

    with st.spinner("Training RNN..."):

        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test,y_test),
            verbose=0
        )


    loss = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )


    st.session_state.trained = True

    st.success("Model Training Completed!")

    st.write(
        "Test MSE:",
        loss
    )


# -------------------------------
# Prediction
# -------------------------------

st.subheader("Predict Close Price")


open_price = st.number_input(
    "Open Price",
    value=1000.0
)


high_price = st.number_input(
    "High Price",
    value=1200.0
)


low_price = st.number_input(
    "Low Price",
    value=900.0
)


volume = st.number_input(
    "Volume",
    value=30000000
)



if st.button("Predict"):

    if not st.session_state.trained:

        st.warning(
            "Please train the model first!"
        )

    else:

        input_data = np.array(
            [[
                open_price,
                high_price,
                low_price,
                volume
            ]]
        )


        input_scaled = scaler_X.transform(
            input_data
        )


        input_rnn = input_scaled.reshape(
            1,
            1,
            4
        )


        prediction = model.predict(
            input_rnn,
            verbose=0
        )


        final_price = scaler_y.inverse_transform(
            prediction
        )


        st.success(
            f"Predicted Close Price: {final_price[0][0]:.2f}"
        )