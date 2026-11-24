"""SuperKart sales forecast - Streamlit front end."""
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_REPO = "cadambank/superkart-sales-model"
MODEL_FILE = "superkart_sales_model_v1.joblib"
FEATURES = ["Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_Type",
            "Product_MRP", "Store_Establishment_Year", "Store_Size",
            "Store_Location_City_Type", "Store_Type"]
PRODUCT_TYPES = ["Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
                 "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
                 "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"]


@st.cache_resource
def load_model():
    """Download the registered model from the Hugging Face model hub (cached once)."""
    return joblib.load(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE))


st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="wide")
st.title("🛒 SuperKart Sales Forecast")
st.write("Predict the total sales revenue of a product in a store from its product and store attributes.")
model = load_model()

single, batch = st.tabs(["Single prediction", "Batch prediction (CSV)"])

with single:
    left, right = st.columns(2)
    with left:
        st.subheader("Product")
        weight = st.number_input("Product weight", min_value=1.0, max_value=30.0, value=12.65, step=0.1)
        sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
        area = st.number_input("Allocated display area (ratio)", min_value=0.0, max_value=0.5,
                               value=0.068, step=0.001, format="%.3f")
        ptype = st.selectbox("Product type", PRODUCT_TYPES, index=PRODUCT_TYPES.index("Fruits and Vegetables"))
        mrp = st.number_input("Product MRP", min_value=1.0, max_value=500.0, value=147.0, step=1.0)
    with right:
        st.subheader("Store")
        year = st.number_input("Store establishment year", min_value=1950, max_value=2030, value=2009, step=1)
        size = st.selectbox("Store size", ["High", "Medium", "Small"], index=1)
        tier = st.selectbox("City type", ["Tier 1", "Tier 2", "Tier 3"], index=1)
        stype = st.selectbox("Store type", ["Supermarket Type1", "Supermarket Type2",
                                            "Departmental Store", "Food Mart"], index=1)

    # Collect the inputs into a single-row DataFrame
    input_df = pd.DataFrame([{
        "Product_Weight": weight, "Product_Sugar_Content": sugar, "Product_Allocated_Area": area,
        "Product_Type": ptype, "Product_MRP": mrp, "Store_Establishment_Year": int(year),
        "Store_Size": size, "Store_Location_City_Type": tier, "Store_Type": stype,
    }])
    st.dataframe(input_df, hide_index=True)

    if st.button("Predict sales", type="primary"):
        prediction = float(model.predict(input_df[FEATURES])[0])
        st.success(f"Forecast Product_Store_Sales_Total: **{prediction:,.2f}**")

with batch:
    st.write("Upload a CSV with these columns: " + ", ".join(f"`{c}`" for c in FEATURES))
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        data = pd.read_csv(file)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data["Predicted_Sales"] = model.predict(data[FEATURES]).round(2)
            st.dataframe(data, hide_index=True)
            st.download_button("Download predictions", data.to_csv(index=False), "predictions.csv", "text/csv")
