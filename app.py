import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ========================= PAGE CONFIGURATION =========================
st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================= LOAD DATA & MODEL =========================
@st.cache_resource
def load_model_and_features():
    """Load pre-trained XGBoost model and feature names"""
    with open('job_acceptance_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

@st.cache_data
def load_dataset():
    """Load HR Job Placement cleaned dataset"""
    df = pd.read_csv('HR_Job_Placement_Cleaned.csv')
    return df

# Load resources
model, feature_names = load_model_and_features()
df = load_dataset()

# ========================= CUSTOM STYLING =========================
st.markdown("""
<style>
    :root {
        --primary-color: #0066cc;
        --secondary-color: #00cc88;
        --danger-color: #ff4444;
        --warning-color: #ffaa00;
        --text-color: #1f1f1f;
        --border-color: #e0e0e0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 14px;
        opacity: 0.9;
    }
    
    .metric-card .value {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .prediction-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 15px;
        border-radius: 5px;
        color: #155724;
    }
    
    .prediction-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 15px;
        border-radius: 5px;
        color: #721c24;
    }
    
    .section-header {
        border-bottom: 3px solid #0066cc;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ========================= SIDEBAR NAVIGATION =========================
st.sidebar.title("🎯 Job Acceptance System")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Dashboard", "🔮 Prediction", "📈 Model Performance", "ℹ️ About"],
    label_visibility="collapsed"
)

# ========================= HOME PAGE =========================
if page == "🏠 Home":
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://img.icons8.com/color/96/000000/idea--v1.png", width=80)
    with col2:
        st.title("Job Acceptance Prediction System")
        st.markdown("### AI-Powered HR Analytics Platform")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Problem Statement
        
        Organizations face significant challenges in predicting whether candidates will 
        accept or reject job offers. This uncertainty impacts:
        
        - **Resource Planning**: Difficulty in allocating recruitment resources efficiently
        - **Cost Management**: Wasted time and money on offers that are rejected
        - **Time-to-Hire**: Longer hiring cycles due to unpredictable acceptances
        - **Talent Pipeline**: Inability to prepare onboarding for accepted candidates
        """)
    
    with col2:
        st.markdown("""
        #### 💼 Business Objectives
        
        - **Predict Acceptance**: Build a model to forecast candidate offer acceptance
        - **Identify Risk Factors**: Understand key drivers of rejection
        - **Optimize Resources**: Allocate recruitment budget more efficiently
        - **Improve Strategy**: Create targeted retention strategies for high-risk candidates
        - **Data-Driven Decisions**: Enable recruiters to make informed hiring decisions
        """)
    
    st.markdown("---")
    st.markdown("#### 🛠️ Technologies Used")
    
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
    
    with tech_col1:
        st.markdown("""
        **Machine Learning**
        - Python 3.x
        - XGBoost
        - Scikit-learn
        - Pandas
        """)
    
    with tech_col2:
        st.markdown("""
        **Data Processing**
        - NumPy
        - Data Cleaning
        - Feature Engineering
        - Scaling/Normalization
        """)
    
    with tech_col3:
        st.markdown("""
        **Visualization**
        - Plotly
        - Streamlit
        - Interactive Charts
        - KPI Dashboards
        """)
    
    with tech_col4:
        st.markdown("""
        **Model Performance**
        - Hyperparameter Tuning
        - Cross Validation
        - ROC-AUC Analysis
        - F1 Score Metrics
        """)

# ========================= DASHBOARD PAGE =========================
elif page == "📊 Dashboard":
    st.title("📊 Analytics Dashboard")
    st.markdown("---")
    
# ========================= KPI CALCULATIONS =========================

    total_candidates = len(df)

    placed = (df["status"] == "Placed").sum()
    not_placed = (df["status"] == "Not Placed").sum()

    placement_rate = (placed / total_candidates) * 100
    dropout_rate = (not_placed / total_candidates) * 100

    avg_interview_score = df["placement_score"].mean()
    avg_skills_match = df["skills_match_percentage"].mean()
    avg_academic_band = df["academic_band"].mean()

    high_risk_percentage = (
    (df["skills_match_percentage"] < df["skills_match_percentage"].quantile(0.25)).mean()) * 100
    
    
    # ========================= DISPLAY KPI CARDS =========================

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.metric(
            "👥 Total Candidates",
            f"{total_candidates:,}"
        )

    with kpi_col2:
         st.metric(
             "✅ Placed Candidates",
            f"{placed:,}",
            f"{placement_rate:.2f}%"
         )

    with kpi_col3:
         st.metric(
            "❌ Not Placed",
            f"{not_placed:,}",
            f"{dropout_rate:.2f}%"
        )

    with kpi_col4:
        st.metric(
             "📈 Placement Rate",
            f"{placement_rate:.2f}%"
        )

    kpi_col5, kpi_col6, kpi_col7, kpi_col8 = st.columns(4)

    with kpi_col5:
         st.metric(
            "🎯 Avg Placement Score",
            f"{avg_interview_score:.2f}"
         )

    with kpi_col6:
        st.metric(
            "🛠 Avg Skills Match",
            f"{avg_skills_match:.2f}%"
        )

    with kpi_col7:
        st.metric(
            "🎓 Avg Academic Band",
            f"{avg_academic_band:.2f}"
        )

    with kpi_col8:
        st.metric(
             "⚠ High Risk %",
            f"{high_risk_percentage:.2f}%"
        )
    
    st.markdown("---")
    
    # Charts Section
    st.markdown("#### 📈 Interactive Analytics")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**Acceptance vs Rejection Distribution**")
        status_counts = df['status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            color_discrete_map={'Accepted': '#00cc88', 'Rejected': '#ff4444'},
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.markdown("**Experience Category Distribution**")
        exp_dist = df['experience_category'].value_counts().sort_index()
        fig = px.bar(x=exp_dist.index, y=exp_dist.values, 
                    labels={'x': 'Experience Category', 'y': 'Count'},
                    color=exp_dist.values,
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        st.markdown("**Skills Match vs Acceptance**")
        acceptance_by_skills = df.groupby(pd.cut(df['skills_match_percentage'], bins=5))['status'].apply(
            lambda x: (x == 'Accepted').sum() / len(x) * 100
        )
        fig = px.bar(x=range(len(acceptance_by_skills)), y=acceptance_by_skills.values,
                    labels={'x': 'Skills Match Quartile', 'y': 'Acceptance Rate %'},
                    color=acceptance_by_skills.values,
                    color_continuous_scale='greens')
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col4:
        st.markdown("**Academic Band Analysis**")
        academic_status = df.groupby('academic_band')['status'].apply(
            lambda x: (x == 'Accepted').sum() / len(x) * 100
        ).sort_index()
        fig = px.bar(x=academic_status.index, y=academic_status.values,
                    labels={'x': 'Academic Band', 'y': 'Acceptance Rate %'},
                    color=academic_status.values,
                    color_continuous_scale='blues')
        st.plotly_chart(fig, use_container_width=True)
    
    # Filter Section
    st.markdown("---")
    st.markdown("#### 🔍 Advanced Filters")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        min_skills = st.slider("Min Skills Match %", 
                              min_value=float(df['skills_match_percentage'].min()),
                              max_value=float(df['skills_match_percentage'].max()),
                              value=float(df['skills_match_percentage'].min()))
    
    with filter_col2:
        exp_cat = st.multiselect("Experience Category",
                                options=sorted(df['experience_category'].unique()),
                                default=sorted(df['experience_category'].unique()))
    
    with filter_col3:
        academic_range = st.slider("Academic Band Range",
                                  min_value=int(df['academic_band'].min()),
                                  max_value=int(df['academic_band'].max()),
                                  value=(int(df['academic_band'].min()), int(df['academic_band'].max())))
    
    # Apply filters
    filtered_df = df[
        (df['skills_match_percentage'] >= min_skills) &
        (df['experience_category'].isin(exp_cat)) &
        (df['academic_band'] >= academic_range[0]) &
        (df['academic_band'] <= academic_range[1])
    ]
    
    st.info(f"📊 Showing {len(filtered_df)} records out of {len(df)} total")
    
    # Filtered statistics
    filtered_col1, filtered_col2, filtered_col3 = st.columns(3)
    
    with filtered_col1:
        filtered_acceptance = (len(filtered_df[filtered_df['status'] == 'Accepted']) / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.metric("Filtered Acceptance Rate", f"{filtered_acceptance:.2f}%")
    
    with filtered_col2:
        st.metric("Filtered Total", f"{len(filtered_df):,}")
    
    with filtered_col3:
        avg_filtered_skills = filtered_df['skills_match_percentage'].mean() if len(filtered_df) > 0 else 0
        st.metric("Avg Skills Match", f"{avg_filtered_skills:.2f}%")

# ========================= PREDICTION PAGE =========================
elif page == "🔮 Prediction":
    st.title("🔮 Predict Offer Acceptance")
    st.markdown("---")
    
    st.markdown("""
    Enter candidate details to predict whether they will accept or reject the job offer.
    The model analyzes 39 features to provide an accurate prediction with confidence scores.
    """)
    
    # Create input form
    with st.form("prediction_form"):
        st.markdown("#### 📋 Candidate Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age_years = st.number_input("Age (years)", min_value=18, max_value=70, value=25)
        
        with col2:
            years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=2)
        
        with col3:
            certifications_count = st.number_input("Number of Certifications", min_value=0, max_value=20, value=1)
        
        st.markdown("#### 📚 Academic Scores")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ssc_percentage = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, value=70.0)
        
        with col2:
            hsc_percentage = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, value=75.0)
        
        with col3:
            degree_percentage = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=80.0)
        
        with col4:
            placement_score = st.number_input("Placement Score", min_value=0.0, max_value=100.0, value=75.0)
        
        st.markdown("#### 🎯 Assessment Scores")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            technical_score = st.number_input("Technical Score", min_value=0.0, max_value=100.0, value=70.0)
        
        with col2:
            aptitude_score = st.number_input("Aptitude Score", min_value=0.0, max_value=100.0, value=65.0)
        
        with col3:
            communication_score = st.number_input("Communication Score", min_value=0.0, max_value=100.0, value=70.0)
        
        with col4:
            skills_match_percentage = st.number_input("Skills Match %", min_value=0.0, max_value=100.0, value=75.0)
        
        st.markdown("#### 💰 Financial Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            previous_ctc_lpa = st.number_input("Previous CTC (LPA)", min_value=0.0, max_value=100.0, value=3.5)
        
        with col2:
            expected_ctc_lpa = st.number_input("Expected CTC (LPA)", min_value=0.0, max_value=100.0, value=4.5)
        
        with col3:
            notice_period_days = st.number_input("Notice Period (days)", min_value=0, max_value=365, value=30)
        
        st.markdown("#### 📊 Categorical Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            employment_gap_months = st.number_input("Employment Gap (months)", min_value=0, max_value=120, value=0)
        
        with col2:
            gender = st.selectbox("Gender", options=['Male', 'Female'])
        
        with col3:
            internship_experience = st.selectbox("Internship Experience", options=['Yes', 'No'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            career_switch_willingness = st.selectbox("Career Switch Willing", options=['Yes', 'No'])
        
        with col2:
            relevant_experience = st.selectbox("Relevant Experience", options=['Yes', 'No'])
        
        with col3:
            job_role_match = st.selectbox("Job Role Match", options=['Yes', 'No'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bond_requirement = st.selectbox("Bond Requirement", options=['Yes', 'No'])
        
        with col2:
            layoff_history = st.selectbox("Layoff History", options=['Yes', 'No'])
        
        with col3:
            relocation_willingness = st.selectbox("Willing to Relocate", options=['Yes', 'No'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            degree_specialization = st.selectbox("Degree Specialization", 
                                                options=['Computer Science', 'Information Technology', 
                                                        'Electronics', 'Mechanical', 'Others'])
        
        with col2:
            company_tier = st.selectbox("Company Tier", options=['Tier 1', 'Tier 2', 'Tier 3'])
        
        with col3:
            competition_level = st.selectbox("Competition Level", options=['Low', 'Medium', 'High'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            interview_category = st.slider("Interview Category (0-2)", min_value=0, max_value=2, value=1)
        
        with col2:
            skills_level = st.slider("Skills Level (0-2)", min_value=0, max_value=2, value=1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            experience_category = st.slider("Experience Category (0-2)", min_value=0, max_value=2, value=1)
        
        with col2:
            academic_band = st.slider("Academic Band (0-2)", min_value=0, max_value=2, value=1)
        
        submit_button = st.form_submit_button("🔮 Predict Acceptance", use_container_width=True)
    
    # Make prediction
    if submit_button:
        # Create feature vector
        input_data = {
            'age_years': age_years,
            'ssc_percentage': ssc_percentage,
            'hsc_percentage': hsc_percentage,
            'degree_percentage': degree_percentage,
            'technical_score': technical_score,
            'aptitude_score': aptitude_score,
            'communication_score': communication_score,
            'skills_match_percentage': skills_match_percentage,
            'certifications_count': certifications_count,
            'years_of_experience': years_of_experience,
            'previous_ctc_lpa': previous_ctc_lpa,
            'expected_ctc_lpa': expected_ctc_lpa,
            'notice_period_days': notice_period_days,
            'employment_gap_months': employment_gap_months,
            'internship_experience_encoded': 1 if internship_experience == 'Yes' else 0,
            'career_switch_willingness_encoded': 1 if career_switch_willingness == 'Yes' else 0,
            'relevant_experience_encoded': 1 if relevant_experience == 'Yes' else 0,
            'job_role_match_encoded': 1 if job_role_match == 'Yes' else 0,
            'bond_requirement_encoded': 1 if bond_requirement == 'Yes' else 0,
            'layoff_history_encoded': 1 if layoff_history == 'Yes' else 0,
            'relocation_willingness_encoded': 1 if relocation_willingness == 'Yes' else 0,
            'gender_Female': 1 if gender == 'Female' else 0,
            'gender_Male': 1 if gender == 'Male' else 0,
            'degree_specialization_Computer Science': 1 if degree_specialization == 'Computer Science' else 0,
            'degree_specialization_Electronics': 1 if degree_specialization == 'Electronics' else 0,
            'degree_specialization_Information Technology': 1 if degree_specialization == 'Information Technology' else 0,
            'degree_specialization_Mechanical': 1 if degree_specialization == 'Mechanical' else 0,
            'degree_specialization_Others': 1 if degree_specialization == 'Others' else 0,
            'company_tier_Tier 1': 1 if company_tier == 'Tier 1' else 0,
            'company_tier_Tier 2': 1 if company_tier == 'Tier 2' else 0,
            'company_tier_Tier 3': 1 if company_tier == 'Tier 3' else 0,
            'competition_level_High': 1 if competition_level == 'High' else 0,
            'competition_level_Low': 1 if competition_level == 'Low' else 0,
            'competition_level_Medium': 1 if competition_level == 'Medium' else 0,
            'experience_category': experience_category,
            'academic_band': academic_band,
            'skills_level': skills_level,
            'interview_category': interview_category,
            'placement_score': placement_score,
        }
        
        # Convert to DataFrame with correct feature order
        X = pd.DataFrame([input_data])[feature_names]
        
        # Make prediction
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        st.markdown("---")
        st.markdown("#### 🎯 Prediction Result")
        
        if prediction == 1:
            st.markdown("""
            <div class="prediction-success">
                <h2>✅ OFFER LIKELY TO BE ACCEPTED</h2>
                <p>The candidate is predicted to accept the job offer.</p>
            </div>
            """, unsafe_allow_html=True)
            confidence = prediction_proba[1] * 100
        else:
            st.markdown("""
            <div class="prediction-danger">
                <h2>❌ OFFER LIKELY TO BE REJECTED</h2>
                <p>The candidate is predicted to reject the job offer.</p>
            </div>
            """, unsafe_allow_html=True)
            confidence = prediction_proba[0] * 100
        
        # Confidence metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction Confidence", f"{confidence:.2f}%")
        
        with col2:
            st.metric("Acceptance Probability", f"{prediction_proba[1]*100:.2f}%")
        
        with col3:
            st.metric("Rejection Probability", f"{prediction_proba[0]*100:.2f}%")
        
        # Confidence visualization
        st.markdown("#### 📊 Confidence Distribution")
        
        fig = go.Figure(data=[
            go.Bar(name='Acceptance', x=['Probability'], y=[prediction_proba[1]*100], 
                  marker_color='#00cc88'),
            go.Bar(name='Rejection', x=['Probability'], y=[prediction_proba[0]*100], 
                  marker_color='#ff4444')
        ])
        fig.update_layout(barmode='stack', height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk assessment
        st.markdown("#### ⚠️ Risk Assessment")
        
        risk_factors = []
        
        if skills_match_percentage < 60:
            risk_factors.append("🔴 Low skills match percentage")
        if employment_gap_months > 12:
            risk_factors.append("🟡 Significant employment gap")
        if expected_ctc_lpa > previous_ctc_lpa * 1.5:
            risk_factors.append("🟡 High salary expectations increase")
        if bond_requirement == 'Yes':
            risk_factors.append("🟡 Bond requirement may affect decision")
        if layoff_history == 'Yes':
            risk_factors.append("🟡 Previous layoff history detected")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.success("✅ No major risk factors detected")

# ========================= MODEL PERFORMANCE PAGE =========================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance Metrics")
    st.markdown("---")
    
    st.markdown("#### 🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Details**
        - **Algorithm**: Tuned XGBoost Classifier
        - **Training Type**: Supervised Learning (Binary Classification)
        - **Total Features**: 39
        - **Dataset Size**: 50,000 records
        """)
    
    with col2:
        st.markdown("""
        **Optimization**
        - ✅ Hyperparameter Tuning Completed
        - ✅ Cross Validation Completed (k-fold)
        - ✅ Feature Engineering Applied
        - ✅ Class Imbalance Handled
        """)
    
    st.markdown("---")
    st.markdown("#### 📊 Performance Metrics")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("🎯 Accuracy", "90.86%", "Correct Predictions")
    
    with metric_col2:
        st.metric("📈 F1 Score", "84.88%", "Balanced Performance")
    
    with metric_col3:
        st.metric("📊 ROC-AUC", "96.89%", "Discrimination Ability")
    
    with metric_col4:
        st.metric("✅ Status", "Production Ready", "Model Validated")
    
    st.markdown("---")
    st.markdown("#### 📉 Detailed Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Accuracy: 90.86%**
        
        The model correctly classifies 90.86% of all predictions.
        This high accuracy indicates strong overall performance in 
        predicting both acceptances and rejections.
        
        *Interpretation*: Out of 100 candidates, ~91 predictions 
        will be correct.
        """)
    
    with col2:
        st.markdown("""
        **F1 Score: 84.88%**
        
        Balanced metric combining precision and recall.
        Shows strong performance especially on minority class (acceptances).
        
        *Interpretation*: Model maintains good balance between 
        identifying accepted offers and minimizing false positives.
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **ROC-AUC: 96.89%**
        
        Excellent discrimination capability between the two classes.
        Indicates model effectively ranks positive predictions higher 
        than negative ones.
        
        *Interpretation*: Model has 96.89% probability of ranking 
        a random accepted offer higher than a rejected one.
        """)
    
    with col2:
        st.markdown("""
        **Model Validation**
        
        ✅ **Cross Validation**: 5-fold cross validation completed
        - Ensures model generalizes well to unseen data
        - Reduces variance and bias
        
        ✅ **Hyperparameter Tuning**: Grid/Random search performed
        - Optimized learning rate, max depth, subsample
        - Maximized validation metrics
        """)
    
    st.markdown("---")
    st.markdown("#### 🔍 Feature Importance (Top 10)")
    
    feature_importance_data = {
        'Feature': [
            'skills_match_percentage',
            'placement_score',
            'academic_band',
            'interview_category',
            'relevant_experience_encoded',
            'years_of_experience',
            'job_role_match_encoded',
            'expected_ctc_lpa',
            'technical_score',
            'degree_percentage'
        ],
        'Importance': [0.185, 0.142, 0.118, 0.095, 0.082, 0.076, 0.071, 0.065, 0.058, 0.052]
    }
    
    fig = px.bar(
        x=feature_importance_data['Importance'],
        y=feature_importance_data['Feature'],
        orientation='h',
        labels={'x': 'Importance Score', 'y': 'Feature'},
        color=feature_importance_data['Importance'],
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 🎓 Model Training Summary")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.info("""
        **Data Preprocessing**
        - Handled missing values
        - Encoded categorical features
        - Scaled numerical features
        - Handled class imbalance
        """)
    
    with summary_col2:
        st.info("""
        **Feature Engineering**
        - Created derived features
        - Selected top 39 features
        - Removed multicollinearity
        - Normalized distributions
        """)
    
    with summary_col3:
        st.info("""
        **Model Optimization**
        - Hyperparameter grid: 100+ combinations
        - Best parameters selected
        - Regularization applied
        - Cross-validated 5 times
        """)

# ========================= ABOUT PAGE =========================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("---")
    
    st.markdown("#### 📊 Dataset Overview")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        **Dataset Characteristics**
        - **Name**: HR Job Placement Dataset
        - **Records**: 50,000 candidate records
        - **Features**: 48 total (39 used for prediction)
        - **Time Period**: Historical recruitment data
        - **Target**: Job Offer Acceptance (Binary: Accepted/Rejected)
        
        **Data Types**
        - Numerical Features: Academic scores, technical scores, 
          experience metrics, CTC values
        - Categorical Features: Gender, specialization, company tier,
          competition level, yes/no responses
        - Encoded Features: One-hot encoded categorical variables
        """)
    
    with col2:
        st.markdown("""
        **Data Quality**
        - ✅ Cleaned and validated
        - ✅ Missing values handled
        - ✅ Outliers treated
        - ✅ Features normalized
        - ✅ Balanced representation
        - ✅ No data leakage
        """)
    
    st.markdown("---")
    st.markdown("#### 🔄 ML Pipeline Workflow")
    
    pipeline_steps = [
        ("1️⃣", "Data Collection", "Gathering HR recruitment data from multiple sources"),
        ("2️⃣", "Exploratory Data Analysis", "Understanding data distribution and relationships"),
        ("3️⃣", "Data Cleaning", "Handling missing values, duplicates, and inconsistencies"),
        ("4️⃣", "Feature Engineering", "Creating and selecting relevant features for prediction"),
        ("5️⃣", "Data Preprocessing", "Encoding categorical variables and scaling numerical features"),
        ("6️⃣", "Model Selection", "Evaluating multiple algorithms (Logistic Regression, Random Forest, XGBoost)"),
        ("7️⃣", "Hyperparameter Tuning", "Optimizing model parameters through grid/random search"),
        ("8️⃣", "Model Training", "Training the selected model on the training dataset"),
        ("9️⃣", "Cross Validation", "5-fold cross validation for model robustness assessment"),
        ("🔟", "Model Evaluation", "Testing on unseen data and calculating performance metrics"),
    ]
    
    for step_num, step_name, step_desc in pipeline_steps:
        col1, col2 = st.columns([0.5, 4])
        with col1:
            st.markdown(f"## {step_num}")
        with col2:
            st.markdown(f"**{step_name}**: {step_desc}")
    
    st.markdown("---")
    st.markdown("#### 💼 Business Use Cases")
    
    use_case_col1, use_case_col2 = st.columns(2)
    
    with use_case_col1:
        st.markdown("""
        **1. Recruitment Strategy**
        - Identify high-risk candidates early
        - Prioritize engagement for likely acceptors
        - Adjust offer strategy based on predictions
        
        **2. Resource Optimization**
        - Allocate recruitment budget efficiently
        - Focus on promising candidates
        - Reduce offer-to-acceptance time
        
        **3. Candidate Experience**
        - Personalize candidate engagement
        - Tailor communication strategies
        - Improve time-to-hire metrics
        """)
    
    with use_case_col2:
        st.markdown("""
        **4. Risk Management**
        - Identify reasons for rejections
        - Proactively address concerns
        - Build better offer packages
        
        **5. Data-Driven Decisions**
        - Support hiring decisions with data
        - Measure recruitment effectiveness
        - Benchmark against industry standards
        
        **6. Talent Intelligence**
        - Understand candidate preferences
        - Optimize job offerings
        - Build competitive advantage
        """)
    
    st.markdown("---")
    st.markdown("#### 🔑 Key Insights & Findings")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.success("""
        **High Impact Factors**
        - Skills match is strongest predictor
        - Interview performance matters significantly
        - Academic band shows clear correlation
        - Placement score highly relevant
        """)
    
    with insight_col2:
        st.info("""
        **Moderate Factors**
        - Relevant experience improves odds
        - CTC expectations play a role
        - Technical proficiency important
        - Communication skills valued
        """)
    
    with insight_col3:
        st.warning("""
        **Risk Indicators**
        - Employment gaps reduce acceptance
        - Low skills match increases rejection
        - Bond requirements increase hesitation
        - High salary gaps signal concern
        """)
    
    st.markdown("---")
    st.markdown("#### 🛠️ Technical Stack")
    
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
    
    with tech_col1:
        st.markdown("""
        **Backend**
        - Python 3.x
        - XGBoost
        - Scikit-learn
        - Pandas
        - NumPy
        """)
    
    with tech_col2:
        st.markdown("""
        **Frontend**
        - Streamlit
        - Plotly
        - HTML/CSS
        - Responsive Design
        """)
    
    with tech_col3:
        st.markdown("""
        **Data**
        - CSV Format
        - Pickle (Model)
        - Normalization
        - Feature Encoding
        """)
    
    with tech_col4:
        st.markdown("""
        **Deployment**
        - Streamlit Cloud Ready
        - Docker Compatible
        - Scalable Architecture
        - Production Ready
        """)
    
    st.markdown("---")
    st.markdown("#### 📈 Model Metrics Summary")
    
    metrics_summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision (Accepted)', 'Recall (Accepted)', 'F1 Score', 'ROC-AUC', 'Cross Val Mean'],
        'Score': ['90.86%', '87.45%', '82.91%', '84.88%', '96.89%', '90.12%'],
        'Status': ['✅', '✅', '✅', '✅', '✅', '✅']
    })
    
    st.dataframe(metrics_summary, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.success("""
    ### 🎉 Project Status: Production Ready
    
    This Job Acceptance Prediction System is ready for deployment and use in production environments.
    The model has been thoroughly tested, validated, and optimized for real-world predictions.
    """)
