# California Housing Price Prediction

A simple web application that predicts California housing prices using machine learning models. The best model is chosen among Decision Tree, Random Forest, AdaBoost, XGBoost, and CatBoost regressors. The model is trained on the California housing dataset and saved using compression for efficient storage.

---

## 🔗 Live Demo

Access the app here:  
➡️ [California Housing Price Predictor](https://all-regresors-pratik-pranav.streamlit.app/)

---

## 📊 Workflow Diagram

```mermaid
flowchart TD
    subgraph Model_Training
        A1[Load California Dataset]
        A2[Preprocess Data]
        A3[Train Multiple Models]
        A4[Select Best Model]
        A5[Save Model as Compressed File]
        
        A1 --> A2 --> A3 --> A4 --> A5
    end

    subgraph App_Workflow
        B1[Create Streamlit App]
        B2[Load Compressed Model File]
        B3[Take User Input]
        B4[Make Prediction]
        B5[Show Prediction Result]
        
        B1 --> B2 --> B3 --> B4 --> B5
    end

    A5 --> B2
