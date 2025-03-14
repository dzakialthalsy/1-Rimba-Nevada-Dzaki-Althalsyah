import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from catboost import CatBoost
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for blue and white theme
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #0066cc;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0066cc;
        color: white;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #004d99;
        color: white;
    }
    h1, h2, h3 {
        color: #0066cc;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        text-align: center;
    }
    .high-risk {
        background-color: #ff4d4d;
        color: white;
    }
    .low-risk {
        background-color: #66cc66;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("Customer Churn Prediction Dashboard")
st.markdown("#### Analyze and predict customer churn using our Machine Learning Model")

# Define feature names - will be used if model doesn't have feature names
feature_names = [
    'current_balance', 
    'average_monthly_balance_prevQ',  
    'current_month_debit',  
    'average_monthly_balance_prevQ2',  
    'previous_month_balance',  
    'previous_month_debit',  
    'current_month_balance',  
    'current_month_credit',  
    'previous_month_end_balance',  
    'age',  
    'customer_nw_category',  
    'previous_month_credit',  
    'occupation_retired'
]

# Load trained model
@st.cache_resource
def load_model():
    try:
        # Try to load the model if it exists
        with open('catboost_model_top.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        # Create a dummy model for demonstration
        st.warning("No model file found. Using a demonstration model.")
        return None

model = load_model()

# Load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('data_prediction.csv')
        return data
    except FileNotFoundError:
        st.error("data_prediction.csv file not found. Please check the file path.")
        # Create sample data for demonstration if file not found
        np.random.seed(42)
        n = 1000
        
        # Create sample data with occupation categories
        occupations = ['Retired', 'Professional', 'Self-employed', 'Salaried', 'Business owner', 'Student', 'Unemployed']
        occupation_probs = [0.15, 0.25, 0.2, 0.2, 0.1, 0.05, 0.05]
        
        sample_data = pd.DataFrame({
            'current_balance': np.random.gamma(5, 1000, n),
            'average_monthly_balance_prevQ': np.random.gamma(5, 950, n),
            'current_month_debit': np.random.gamma(2, 600, n),
            'average_monthly_balance_prevQ2': np.random.gamma(5, 900, n),
            'previous_month_balance': np.random.gamma(5, 980, n),
            'previous_month_debit': np.random.gamma(2, 550, n),
            'current_month_balance': np.random.gamma(5, 1020, n),
            'current_month_credit': np.random.gamma(3, 700, n),
            'previous_month_end_balance': np.random.gamma(5, 970, n),
            'age': np.random.randint(18, 80, n),
            'customer_nw_category': np.random.choice([1, 2, 3], size=n, p=[0.3, 0.5, 0.2]),
            'previous_month_credit': np.random.gamma(3, 650, n),
            'occupation_retired': np.random.randint(0, 2, n),
            'occupation': np.random.choice(occupations, size=n, p=occupation_probs),
            'churn': np.random.choice([0, 1], size=n, p=[0.8, 0.2])
        })
        
        # Making occupation_retired consistent with occupation column
        sample_data.loc[sample_data['occupation'] == 'Retired', 'occupation_retired'] = 1
        
        return sample_data

# Load the feature importance data from model
@st.cache_data
def get_feature_importance(_model):
    try:
        # Try to get feature importance from the model
        importance = _model.get_feature_importance()
        
        # Get feature names from the model if available
        try:
            model_feature_names = _model.feature_names_
            # If model feature names is None, use our predefined list
            if model_feature_names is None:
                model_feature_names = feature_names
        except (AttributeError, TypeError):
            # If model doesn't have feature_names_ attribute, use our predefined list
            model_feature_names = feature_names
        
        # Make sure lengths match
        if len(model_feature_names) != len(importance):
            st.warning(f"Feature names length ({len(model_feature_names)}) doesn't match importance length ({len(importance)}). Using sample values.")
            # Use sample values with correct length
            importance = np.linspace(0.15, 0.01, len(model_feature_names))
        
        importance_df = pd.DataFrame({
            'Feature': model_feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    except Exception as e:
        # If there's an error or model doesn't have feature importance
        st.warning(f"Could not extract feature importance from model: {e}. Using sample values.")
        # Generate sample importance values with the correct length
        importance = np.linspace(0.15, 0.01, len(feature_names))
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

# Load the data
data = load_data()

# Create tabs
tab1, tab2 = st.tabs(["📊 Churn Prediction", "🔍 Model Insights"])

# Tab 1: Churn Prediction
with tab1:
    st.header("Customer Churn Prediction")
    st.markdown("Enter customer details to predict churn probability:")
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        current_balance = st.number_input("Current Balance", min_value=0.0, value=5000.0, step=100.0)
        avg_balance_prevQ = st.number_input("Average Monthly Balance Previous Quarter", min_value=0.0, value=4800.0, step=100.0)
        current_month_debit = st.number_input("Current Month Debit", min_value=0.0, value=1200.0, step=100.0)
        avg_balance_prevQ2 = st.number_input("Average Monthly Balance 2 Quarters Ago", min_value=0.0, value=4600.0, step=100.0)
        previous_month_balance = st.number_input("Previous Month Balance", min_value=0.0, value=4900.0, step=100.0)
        previous_month_debit = st.number_input("Previous Month Debit", min_value=0.0, value=1100.0, step=100.0)
    
    with col2:
        current_month_balance = st.number_input("Current Month Balance", min_value=0.0, value=5100.0, step=100.0)
        current_month_credit = st.number_input("Current Month Credit", min_value=0.0, value=2000.0, step=100.0)
        previous_month_end_balance = st.number_input("Previous Month End Balance", min_value=0.0, value=4850.0, step=100.0)
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        customer_nw_category = st.selectbox("Customer Net Worth Category", options=["Low", "Medium", "High"])
        previous_month_credit = st.number_input("Previous Month Credit", min_value=0.0, value=1950.0, step=100.0)
        occupation_retired = st.checkbox("Retired")
    
    # Mapping for categorical variables
    customer_nw_category_map = {"Low": 1, "Medium": 2, "High": 3}
    
    # Convert categorical to numerical
    customer_nw_category_val = customer_nw_category_map[customer_nw_category]
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'current_balance': [current_balance],
        'average_monthly_balance_prevQ': [avg_balance_prevQ],
        'current_month_debit': [current_month_debit],
        'average_monthly_balance_prevQ2': [avg_balance_prevQ2],
        'previous_month_balance': [previous_month_balance],
        'previous_month_debit': [previous_month_debit],
        'current_month_balance': [current_month_balance],
        'current_month_credit': [current_month_credit],
        'previous_month_end_balance': [previous_month_end_balance],
        'age': [age],
        'customer_nw_category': [customer_nw_category_val],
        'previous_month_credit': [previous_month_credit],
        'occupation_retired': [1 if occupation_retired else 0]
    })
    threshold = 0.6732070360990345
    # Prediction button
    if st.button("Predict Churn Probability"):
        try:
            if model is None:
                # Generate a random prediction if no model is available
                prediction = np.random.random()
                st.warning("Using a demo prediction as no model was loaded.")
            else:
                prediction = model.predict_proba(input_data)
            
                # Check the shape of prediction output - CatBoost might return different formats
                if hasattr(prediction, 'shape') and len(prediction.shape) > 1 and prediction.shape[1] > 1:
                    # It's returning a 2D array with probabilities for both classes
                    prediction = prediction[0, 1]  # Get probability of positive class (index 1)
                else:
                    # It's returning just the positive class probability
                    prediction = prediction[0]
            
            # Apply the threshold to classify churn
            churn_pred = 1 if prediction >= threshold else 0
            
            # Display prediction results
            st.markdown("### Prediction Results")
            st.write("Predicted churn:" + (" Yes" if churn_pred == 1 else " No"))
            
            # Create a gauge chart for the churn probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': 'green'},
                        {'range': [30, 70], 'color': 'yellow'},
                        {'range': [70, 100], 'color': 'red'}
                    ],
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk assessment
            if prediction > 0.5:
                st.markdown(f"""
                <div class="prediction-box high-risk">
                    <h3>High Churn Risk: {prediction:.1%}</h3>
                    <p>This customer is likely to churn. Consider implementing retention strategies.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Retention suggestions
                st.subheader("Recommended Retention Actions:")
                actions = [
                    "Offer personalized loyalty rewards",
                    "Contact customer for feedback",
                    "Provide special promotional rates",
                    "Schedule account review with relationship manager"
                ]
                for action in actions:
                    st.markdown(f"- {action}")
                
            else:
                st.markdown(f"""
                <div class="prediction-box low-risk">
                    <h3>Low Churn Risk: {prediction:.1%}</h3>
                    <p>This customer is likely to stay. Continue providing excellent service.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Engagement suggestions
                st.subheader("Recommended Engagement Actions:")
                actions = [
                    "Offer additional product suggestions",
                    "Regular service quality check-ins",
                    "Invite to loyalty program if not enrolled",
                    "Share educational resources on financial planning"
                ]
                for action in actions:
                    st.markdown(f"- {action}")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Tab 2: Model Insights
with tab2:
    st.header("Model Insights & Feature Importance")
    
    # Get feature importance
    feature_imp_df = get_feature_importance(model)
    
    # Create columns for visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Feature importance plot
        st.subheader("Feature Importance")
        
        fig = px.bar(
            feature_imp_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues',
            labels={'Importance': 'Relative Importance', 'Feature': ''},
            height=500
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=10, r=10, t=10, b=10),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Factors")
        top_features = feature_imp_df.head(5)
        
        for i, (_, row) in enumerate(top_features.iterrows()):
            st.markdown(f"""
            <div style="padding: 10px; background-color: rgba(0, 102, 204, {0.9 - i*0.15}); color: white; border-radius: 5px; margin-bottom: 10px;">
                <b>{row['Feature']}</b>: {row['Importance']:.3f}
            </div>
            """, unsafe_allow_html=True)
    
    # Data visualizations using actual data
    if not data.empty:
        st.header("Data Visualizations")
        
        # Check if 'churn' column exists in the data
        if 'churn' not in data.columns:
            st.warning("The 'churn' column is missing from your dataset. Some visualizations may not display correctly.")
        
        # Create tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Balance Analysis", "Age Distribution", "Credit-Debit Patterns", "Churn Analysis"])
        
        with viz_tab1:
            st.subheader("Current Balance vs Previous Month Balance by Churn Status")
            
            # Check if required columns exist
            required_columns = ['current_balance', 'previous_month_balance', 'churn']
            if all(col in data.columns for col in required_columns):
                fig = px.scatter(
                    data,
                    x="current_balance",
                    y="previous_month_balance",
                    color="churn",
                    color_discrete_map={0: "#0066cc", 1: "#ff4d4d"},
                    labels={"current_balance": "Current Balance", "previous_month_balance": "Previous Month Balance", "churn": "Churned"},
                    hover_data=["age", "customer_nw_category"] if "age" in data.columns and "customer_nw_category" in data.columns else None,
                    opacity=0.7,
                    size_max=10,
                    title="Balance Comparison by Churn Status"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Insight**: Customers with significant drops in balance between previous month and current month 
                show higher churn rates. This could indicate dissatisfaction or account closing preparation.
                """)
            else:
                st.warning("Missing required columns for this visualization.")
        
        with viz_tab2:
            st.subheader("Age Distribution by Churn Status")
            
            # Check if required columns exist
            required_columns = ['age', 'churn']
            if all(col in data.columns for col in required_columns):
                fig = px.histogram(
                    data,
                    x="age",
                    color="churn",
                    barmode="overlay",
                    color_discrete_map={0: "#0066cc", 1: "#ff4d4d"},
                    labels={"age": "Customer Age", "churn": "Churned"},
                    opacity=0.7,
                    nbins=20,
                )
                
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate churn rate by age group
                data_copy = data.copy()
                data_copy['age_group'] = pd.cut(data_copy['age'], bins=[18, 30, 40, 50, 60, 80], labels=['18-30', '31-40', '41-50', '51-60', '61+'])
                age_churn = data_copy.groupby('age_group')['churn'].mean().reset_index()
                age_churn['churn_pct'] = age_churn['churn'] * 100
                
                fig = px.bar(
                    age_churn,
                    x='age_group',
                    y='churn_pct',
                    color='churn_pct',
                    color_continuous_scale='Blues_r',
                    labels={'age_group': 'Age Group', 'churn_pct': 'Churn Rate (%)'},
                    title="Churn Rate by Age Group"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Insight**: Different age groups show varying churn patterns. Targeted retention strategies 
                should be developed for high-risk age segments.
                """)
            else:
                st.warning("Missing required columns for this visualization.")
        
        with viz_tab3:
            st.subheader("Credit-Debit Patterns by Churn Status")
            
            # Check if required columns exist
            required_columns = ['current_month_credit', 'current_month_debit', 
                               'previous_month_credit', 'previous_month_debit', 'churn']
            if all(col in data.columns for col in required_columns):
                # Create aggregated data for current vs previous month activity
                data_copy = data.copy()
                data_copy['credit_debit_ratio_current'] = data_copy['current_month_credit'] / (data_copy['current_month_debit'] + 1)  # Avoid division by zero
                data_copy['credit_debit_ratio_prev'] = data_copy['previous_month_credit'] / (data_copy['previous_month_debit'] + 1)
                
                fig = px.scatter(
                    data_copy,
                    x="credit_debit_ratio_prev",
                    y="credit_debit_ratio_current",
                    color="churn",
                    color_discrete_map={0: "#0066cc", 1: "#ff4d4d"},
                    labels={
                        "credit_debit_ratio_prev": "Previous Month Credit/Debit Ratio", 
                        "credit_debit_ratio_current": "Current Month Credit/Debit Ratio", 
                        "churn": "Churned"
                    },
                    opacity=0.7,
                    title="Change in Credit-Debit Ratio"
                )
                
                # Add reference line for y=x
                fig.add_trace(
                    go.Scatter(
                        x=[0, 5],
                        y=[0, 5],
                        mode="lines",
                        line=dict(color="gray", dash="dash"),
                        name="No Change Line"
                    )
                )
                
                fig.update_layout(
                    xaxis=dict(range=[0, 5]),
                    yaxis=dict(range=[0, 5])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Insight**: Customers below the "No Change Line" have a declining credit-to-debit ratio, 
                potentially indicating reduced account activity or preparation for account closure. 
                These customers show higher churn rates.
                """)
            else:
                st.warning("Missing required columns for this visualization.")
        
        # New tab for Churn Analysis with Occupation Chart and Pie Chart
        with viz_tab4:
            st.subheader("Churn Analysis")
            
            # Create two columns for the charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                try:
                    # Try to load the churn dataset
                    data2 = pd.read_csv('churn_dataset_clean.csv')
                    
                    # Check if occupation column exists
                    if 'occupation' in data2.columns and 'churn' in data2.columns:
                        # Calculate churn rate by occupation
                        occupation_churn = data2.groupby('occupation')['churn'].mean().reset_index()
                        occupation_churn['churn_pct'] = occupation_churn['churn'] * 100
                        occupation_churn = occupation_churn.sort_values('churn_pct', ascending=False)
                        
                        # Create bar chart for churn by occupation
                        fig = px.bar(
                            occupation_churn,
                            x='occupation',
                            y='churn_pct',
                            color='churn_pct',
                            color_continuous_scale='Blues',
                            labels={'occupation': 'Occupation', 'churn_pct': 'Churn Rate (%)'},
                            title="Churn Rate by Occupation"
                        )
                        
                        fig.update_layout(
                            xaxis={'categoryorder': 'total descending'},
                            yaxis_title="Churn Rate (%)"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Find highest and lowest churn occupations
                        highest_churn_occ = occupation_churn.iloc[0]['occupation']
                        highest_churn_rate = occupation_churn.iloc[0]['churn_pct']
                        lowest_churn_occ = occupation_churn.iloc[-1]['occupation']
                        lowest_churn_rate = occupation_churn.iloc[-1]['churn_pct']
                        
                        st.markdown(f"""
                        **Insight**: {highest_churn_occ} customers have the highest churn rate at {highest_churn_rate:.1f}%, 
                        while {lowest_churn_occ} customers show the lowest at {lowest_churn_rate:.1f}%. This suggests 
                        that occupation-specific strategies may be effective for retention.
                        """)
                    else:
                        raise ValueError("Missing occupation or churn columns in churn_dataset_clean.csv")
                        
                except Exception as e:
                    # Fall back to current dataset if churn_dataset_clean.csv has issues
                    st.warning(f"Could not load churn_dataset_clean.csv: {e}. Using main dataset for visualization.")
                    
                    # Check if occupation_retired exists in main dataset
                    if 'occupation_retired' in data.columns and 'churn' in data.columns:
                        # If only occupation_retired exists, create a simplified version
                        retired_churn = data.groupby('occupation_retired')['churn'].mean().reset_index()
                        retired_churn['occupation'] = retired_churn['occupation_retired'].map({1: 'Retired', 0: 'Not Retired'})
                        retired_churn['churn_pct'] = retired_churn['churn'] * 100
                        
                        fig = px.bar(
                            retired_churn,
                            x='occupation',
                            y='churn_pct',
                            color='churn_pct',
                            color_continuous_scale='Blues_r',
                            labels={'occupation': 'Retirement Status', 'churn_pct': 'Churn Rate (%)'},
                            title="Churn Rate by Retirement Status"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Compare retired vs not retired
                        retired_rate = retired_churn[retired_churn['occupation_retired'] == 1]['churn_pct'].values[0]
                        not_retired_rate = retired_churn[retired_churn['occupation_retired'] == 0]['churn_pct'].values[0]
                        
                        st.markdown(f"""
                        **Insight**: Retired customers have a churn rate of {retired_rate:.1f}%, compared to 
                        {not_retired_rate:.1f}% for non-retired customers. This suggests that 
                        {"retired customers require special attention" if retired_rate > not_retired_rate else "non-retired customers require special attention"}.
                        """)
                    else:
                        st.warning("Missing occupation data for this visualization.")
            
            with chart_col2:
                # Check if churn column exists
                if 'churn' in data.columns:
                    # Calculate overall churn distribution
                    churn_counts = data['churn'].value_counts().reset_index()
                    churn_counts.columns = ['Status', 'Count']
                    churn_counts['Status'] = churn_counts['Status'].map({0: 'Retained', 1: 'Churned'})
                    
                    # Create pie chart for churn distribution
                    fig = px.pie(
                        churn_counts,
                        values='Count',
                        names='Status',
                        color='Status',
                        color_discrete_map={'Retained': '#0066cc', 'Churned': '#ff4d4d'},
                        title="Overall Churn Distribution"
                    )
                    
                    fig.update_layout(
                        legend_title_text='Customer Status'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate overall churn percentage
                    total_customers = churn_counts['Count'].sum()
                    churn_percentage = churn_counts[churn_counts['Status'] == 'Churned']['Count'].values[0] / total_customers * 100
                    
                    st.markdown(f"""
                    **Insight**: The overall churn rate is {churn_percentage:.1f}% of the total customer base. 
                    This provides a baseline for evaluating the effectiveness of retention strategies across 
                    different customer segments.
                    """)
                else:
                    st.warning("Missing churn data for this visualization.")
    else:
        st.warning("No data available for visualizations. Please check your data file.")

# Sidebar with additional information
try:
    st.sidebar.image("icon.png", width=100)
except:
    st.sidebar.info("Icon image not found.")

st.sidebar.title("About This Tool")
st.sidebar.markdown("""
This dashboard helps predict customer churn using machine learning.

**Instructions:**
1. Enter customer financial data in the "Churn Prediction" tab
2. Click "Predict Churn Probability" to see results
3. Explore model insights and data patterns in the "Model Insights" tab

**Features Used:**
""")

for feature in feature_names:
    st.sidebar.markdown(f"- {feature.replace('_', ' ').title()}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Hyperparameter Tuned CatBoost Classifier")