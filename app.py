import streamlit as st
import pandas as pd
import pickle



# ==================
# Load files
# ==================


with open(
    "models/scaler.pkl",
    "rb"
) as f:

    scaler = pickle.load(f)



with open(
    "models/predictor.pkl",
    "rb"
) as f:

    model = pickle.load(f)




st.title(
    "Iris Clustering using t-SNE"
)



# Inputs


sepal_length = st.number_input(
    "Sepal Length",
    value=5.1
)


sepal_width = st.number_input(
    "Sepal Width",
    value=3.5
)


petal_length = st.number_input(
    "Petal Length",
    value=1.4
)


petal_width = st.number_input(
    "Petal Width",
    value=0.2
)



# ==================
# Feature Engineering
# ==================


sepal_ratio = (
    sepal_length /
    sepal_width
)


petal_ratio = (
    petal_length /
    petal_width
)



data = pd.DataFrame(
    [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width,
        sepal_ratio,
        petal_ratio
    ]],
    columns=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
        "Sepal_Ratio",
        "Petal_Ratio"
    ]
)



# Prediction


if st.button(
    "Predict Cluster"
):


    scaled = scaler.transform(
        data
    )


    result = model.predict(
        scaled
    )


    st.success(
        f"Flower belongs to Cluster {result[0]}"
    )



    if result[0] == 0:

        st.info(
            "Iris Group 1"
        )


    elif result[0] == 1:

        st.info(
            "Iris Group 2"
        )


    else:

        st.info(
            "Iris Group 3"
        )