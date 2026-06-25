import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------

st.markdown(
    """
    <style>

    .main-header{
        font-size:3rem;
        color:#6a0dad;
        text-align:center;
        margin-bottom:10px;
        padding-bottom:20px;
    }

    .prediction-card{
        background-color:#f0f8ff;
        padding:2rem;
        border-radius:10px;
        border-left:5px solid #6a0dad;
        margin:1rem 0;
    }

    .confidence-bar{
        height:20px;
        background-color:#e0e0e0;
        border-radius:10px;
        margin:0.5rem 0;
    }

    .confidence-fill{
        height:100%;
        border-radius:10px;
        background:linear-gradient(90deg,#ff6b6b,#4ecdc4);
        text-align:center;
        color:white;
        font-weight:bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

@st.cache_resource
def load_model(model_format="joblib"):

    try:

        if model_format == "joblib":

            model = joblib.load(
                "models/iris_model.joblib"
            )

        else:

            with open(
                "models/iris_model.pickle",
                "rb"
            ) as f:

                model = pickle.load(f)

        return model

    except Exception as e:

        st.error(
            f"Error loading model: {e}"
        )

        return None


# ----------------------------------------------------
# LOAD MODEL INFO
# ----------------------------------------------------

@st.cache_resource
def load_model_info():

    try:

        with open(
            "models/model_info.json",
            "r"
        ) as f:

            return json.load(f)

    except Exception as e:

        st.error(
            f"Error loading model info: {e}"
        )

        return None


# ----------------------------------------------------
# LOAD FEATURE RANGES
# ----------------------------------------------------

@st.cache_resource
def load_feature_ranges():

    try:

        with open(
            "models/feature_ranges.json",
            "r"
        ) as f:

            return json.load(f)

    except Exception:

        return {

            "sepal_length": {
                "min": 4.0,
                "max": 8.0,
                "default": 5.8
            },

            "sepal_width": {
                "min": 2.0,
                "max": 4.5,
                "default": 3.0
            },

            "petal_length": {
                "min": 1.0,
                "max": 7.0,
                "default": 4.0
            },

            "petal_width": {
                "min": 0.1,
                "max": 2.5,
                "default": 1.2
            }
        }


# ----------------------------------------------------
# LOAD EVERYTHING
# ----------------------------------------------------

model_info = load_model_info()

feature_ranges = load_feature_ranges()

model = load_model("joblib")

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

with st.sidebar:

    st.title("⚙️ Settings")

    # Model selection
    model_format = st.radio(
        "Model Format",
        ["joblib", "pickle"],
        help="Choose which model format to use"
    )

    # Reload model button
    if st.button(" Reload Model"):

        model = load_model(model_format)

        if model is not None:

            st.success(
                f"Loaded {model_format} model successfully!"
            )

    st.divider()

    # ------------------------------------------------
    # MODEL INFORMATION
    # ------------------------------------------------

    st.subheader(" Model Information")

    if model_info is not None:

        st.write(
            f"**Model Type:** "
            f"{model_info.get('model_type')}"
        )

        st.write(
            f"**Accuracy:** "
            f"{model_info.get('accuracy',0):.2%}"
        )

        st.write(
            f"**Features:** "
            f"{len(model_info.get('feature_names',[]))}"
        )

        st.write(
            f"**Classes:** "
            f"{len(model_info.get('target_names',[]))}"
        )

    st.divider()




# ----------------------------------------------------
# MAIN HEADER
# ----------------------------------------------------

st.markdown(
    """
    <h1 class="main-header">
    🌸 Iris Flower Classification
    </h1>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# DESCRIPTION
# ----------------------------------------------------

st.markdown(
    """
This app predicts the species of an Iris flower based on its measurements.

Adjust the sliders below and click **Predict Species** to see the prediction and confidence scores.
"""
)

# ----------------------------------------------------
# TWO-COLUMN LAYOUT
# ----------------------------------------------------

col1, col2 = st.columns([2, 1])

# ----------------------------------------------------
# INPUT SECTION
# ----------------------------------------------------

with col1:

    st.header(" Input Features")

    sepal_length = st.slider(
        "Sepal Length (cm)",
        min_value=float(feature_ranges["sepal_length"]["min"]),
        max_value=float(feature_ranges["sepal_length"]["max"]),
        value=float(feature_ranges["sepal_length"]["default"]),
        step=0.1
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        min_value=float(feature_ranges["sepal_width"]["min"]),
        max_value=float(feature_ranges["sepal_width"]["max"]),
        value=float(feature_ranges["sepal_width"]["default"]),
        step=0.1
    )

    petal_length = st.slider(
        "Petal Length (cm)",
        min_value=float(feature_ranges["petal_length"]["min"]),
        max_value=float(feature_ranges["petal_length"]["max"]),
        value=float(feature_ranges["petal_length"]["default"]),
        step=0.1
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        min_value=float(feature_ranges["petal_width"]["min"]),
        max_value=float(feature_ranges["petal_width"]["max"]),
        value=float(feature_ranges["petal_width"]["default"]),
        step=0.1
    )


# ----------------------------------------------------
# CURRENT VALUES TABLE
# ----------------------------------------------------

with col2:

    st.header(" Current Values")

    features_df = pd.DataFrame(
        {
            "Feature": [
                "Sepal Length",
                "Sepal Width",
                "Petal Length",
                "Petal Width"
            ],

            "Value (cm)": [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]
        }
    )

    st.dataframe(
        features_df,
        hide_index=True,
        use_container_width=True
    )


# ----------------------------------------------------
# CREATE INPUT ARRAY
# ----------------------------------------------------

input_features = np.array(
    [
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ]
)

# ----------------------------------------------------
# PREDICTION BUTTON
# ----------------------------------------------------

if st.button(
    " Predict Species",
    type="primary",
    use_container_width=True
):

    if model is not None and model_info is not None:

        try:

            # Make prediction
            prediction = model.predict(
                input_features
            )

            prediction_proba = model.predict_proba(
                input_features
            )[0]

            # Predicted class name
            predicted_class = (
                model_info["target_names"][
                    prediction[0]
                ]
            )

            # ----------------------------------------
            # RESULT CARD
            # ----------------------------------------

            st.markdown(
                '<div class="prediction-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "##  Prediction Result"
            )

            st.success(
                f"Predicted Species: {predicted_class}"
            )

            # ----------------------------------------
            # CONFIDENCE SCORES
            # ----------------------------------------

            st.subheader(
                " Confidence Scores"
            )

            for i, probability in enumerate(
                prediction_proba
            ):

                species = (
                    model_info["target_names"][i]
                )

                percentage = probability * 100

                st.write(
                    f"**{species}**"
                )

                st.progress(
                    probability
                )

                st.write(
                    f"{percentage:.2f}%"
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(
                f"Error while making prediction:\n{e}"
            )

    else:

        st.error(
            "Model could not be loaded."
        )


# ----------------------------------------------------
# DATASET INFORMATION
# ----------------------------------------------------

with st.expander(
    " About the Iris Dataset"
):

    st.markdown(
        """
### Dataset Characteristics

- 150 samples
- 50 samples per class
- 4 features
- 3 flower species

### Species

- Iris Setosa
- Iris Versicolor
- Iris Virginica

### Features

1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

### Model

Random Forest Classifier
"""
    )


# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------

st.markdown("---")

st.markdown(
    """
<div style='text-align:center'>
<p>
Built with Streamlit and Scikit-Learn
</p>
</div>
""",
    unsafe_allow_html=True
)
