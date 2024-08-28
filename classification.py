import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    return df, iris.target_names


df, target_names = load_data()
st.write("## The first 10 rows of the dataset are :", df.head(10))


X_train, X_test, y_train, y_test = train_test_split(
    df.drop("species", axis=1), df["species"], test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"### Model Accuracy: {100*accuracy:.2f} %")

st.sidebar.title("Input Features")
sepal_length = st.sidebar.slider(
    "Sepal Length",
    float(df["sepal length (cm)"].min()),
    float(df["sepal length (cm)"].max()),
)
sepal_width = st.sidebar.slider(
    "Sepal Width",
    float(df["sepal width (cm)"].min()),
    float(df["sepal width (cm)"].max()),
)
petal_length = st.sidebar.slider(
    "Petal Length",
    float(df["petal length (cm)"].min()),
    float(df["petal length (cm)"].max()),
)
petal_width = st.sidebar.slider(
    "Petal Width",
    float(df["petal width (cm)"].min()),
    float(df["petal width (cm)"].max()),
)

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
ip_data = pd.DataFrame(input_data, columns=df.columns[:-1])
prediction = model.predict(ip_data)
predicted_species = target_names[prediction[0]]

st.write(f"## The Species belong to:\n # {predicted_species}")
