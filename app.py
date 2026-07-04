import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration – no emoji, clean design
st.set_page_config(
    page_title="Bank Customer Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "Bank Customer Analytics Dashboard"
    }
)

# ----
# Data generation and loading
# ----

@st.cache_data
def generate_sample_data(n_samples=5000):
    """Generate realistic sample bank customer data."""
    np.random.seed(42)
    
    # Demographics
    ages = np.random.normal(40, 12, n_samples).astype(int)
    ages = np.clip(ages, 18, 80)
    
    jobs = np.random.choice([
        'admin', 'technician', 'services', 'management', 
        'retired', 'blue-collar', 'unemployed', 'entrepreneur',
        'housemaid', 'student', 'self-employed'
    ], n_samples, p=[0.15, 0.12, 0.10, 0.08, 0.08, 0.12, 0.05, 0.06, 0.04, 0.08, 0.12])
    
    marital = np.random.choice(['married', 'single', 'divorced'], n_samples, p=[0.6, 0.3, 0.1])
    
    education = np.random.choice([
        'primary', 'secondary', 'tertiary', 'unknown'
    ], n_samples, p=[0.2, 0.5, 0.25, 0.05])
    
    # Financial data
    balances = np.random.lognormal(8, 1.5, n_samples)
    balances = np.clip(balances, 0, 200000)
    
    durations = np.random.exponential(180, n_samples).astype(int)
    durations = np.clip(durations, 30, 1200)
    
    loan_probs = np.where(balances > 10000, 0.3, 0.1)
    loans = np.random.binomial(1, loan_probs, n_samples)
    loan_status = ['yes' if x else 'no' for x in loans]
    
    df = pd.DataFrame({
        'age': ages,
        'job': jobs,
        'marital': marital,
        'education': education,
        'balance': balances,
        'loan': loan_status,
        'duration': durations
    })
    return df


@st.cache_data
def load_data():
    """Load data with fallback to generated sample data."""
    with st.spinner('Loading and processing data...'):
        try:
            df = pd.read_excel('cleaned_bank_data_copy.xlsx')
            st.success("Loaded data from Excel file")
        except FileNotFoundError:
            df = generate_sample_data(5000)
            st.info("Using generated sample data (original file not found)")
        except Exception as e:
            df = generate_sample_data(5000)
            st.warning(f"Error loading file: {str(e)}. Using sample data instead.")
        
        # Data cleaning
        df = df.replace('unknown', np.nan)
        df = df.drop_duplicates()
        
        # Ensure non‑negative numeric values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].apply(lambda x: max(x, 0) if pd.notna(x) else x)
        
        # Derived features
        bins = [0, 18, 25, 35, 45, 55, 65, 75, 100]
        labels = ['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-100']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
        
        balance_bins = [0, 1000, 5000, 10000, 25000, 50000, 100000, float('inf')]
        balance_labels = ['<$1K', '$1K-5K', '$5K-10K', '$10K-25K', '$25K-50K', '$50K-100K', '>$100K']
        df['balance_category'] = pd.cut(df['balance'], bins=balance_bins, labels=balance_labels)
        
        duration_bins = [0, 60, 120, 180, 240, 300, 600, float('inf')]
        duration_labels = ['<1min', '1-2min', '2-3min', '3-4min', '4-5min', '5-10min', '>10min']
        df['duration_category'] = pd.cut(df['duration'], bins=duration_bins, labels=duration_labels)
        
        # Synthetic date/time features for trend analysis
        start_date = datetime.now() - timedelta(days=365*2)
        dates = [start_date + timedelta(days=np.random.randint(0, 365*2)) for _ in range(len(df))]
        df['date'] = pd.to_datetime(dates)
        df['month'] = df['date'].dt.to_period('M').astype(str)
        df['quarter'] = df['date'].dt.to_period('Q').astype(str)
        df['month_name'] = df['date'].dt.month_name()
        
        # Synthetic churn risk
        df['churn_risk'] = np.random.choice(['Low', 'Medium', 'High'], size=len(df), p=[0.7, 0.2, 0.1])
        
    return df


# ----
# Load data
# ----
df = load_data()

# Set base style for static plots (not critical with Plotly, but kept for seaborn)
sns.set_style("whitegrid")
plt.style.use('default')

# ----
# Sidebar filters
# ----
with st.sidebar:
    st.header("Filters")
    
    if st.button("Reset All Filters", help="Reset all filters to default values"):
        st.rerun()
    
    with st.expander("Demographic Filters", expanded=True):
        age_range = st.slider(
            "Select Age Range:",
            min_value=int(df['age'].min()),
            max_value=int(df['age'].max()),
            value=(int(df['age'].min()), int(df['age'].max())),
            help="Filter customers by age range"
        )
        
        job_filter = st.multiselect(
            "Select Job Types:",
            options=sorted(df['job'].dropna().unique()),
            default=sorted(df['job'].dropna().unique()),
            help="Filter by customer occupation"
        )
        
        marital_filter = st.multiselect(
            "Select Marital Status:",
            options=sorted(df['marital'].dropna().unique()),
            default=sorted(df['marital'].dropna().unique()),
            help="Filter by marital status"
        )
        
        education_filter = st.multiselect(
            "Select Education Level:",
            options=sorted(df['education'].dropna().unique()),
            default=sorted(df['education'].dropna().unique()),
            help="Filter by education level"
        )
    
    with st.expander("Financial Filters"):
        balance_range = st.slider(
            "Select Balance Range ($):",
            min_value=int(df['balance'].min()),
            max_value=int(df['balance'].max()),
            value=(int(df['balance'].min()), min(50000, int(df['balance'].max()))),
            step=1000,
            help="Filter by account balance range"
        )
        
        loan_filter = st.multiselect(
            "Select Loan Status:",
            options=sorted(df['loan'].dropna().unique()),
            default=sorted(df['loan'].dropna().unique()),
            help="Filter by loan status"
        )
    
    with st.expander("Engagement Filters"):
        duration_range = st.slider(
            "Select Call Duration Range (sec):",
            min_value=int(df['duration'].min()),
            max_value=int(df['duration'].max()),
            value=(int(df['duration'].min()), min(300, int(df['duration'].max()))),
            help="Filter by call duration"
        )
    
    with st.expander("Advanced Options"):
        sample_size = st.slider(
            "Sample Size (for performance):",
            min_value=100,
            max_value=len(df),
            value=min(5000, len(df)),
            step=100,
            help="Reduce sample size for better performance"
        )
        
        show_outliers = st.checkbox(
            "Show Outliers in Plots",
            value=True,
            help="Toggle outlier visibility in box plots"
        )
        
        color_palette = st.selectbox(
            "Select Color Palette:",
            options=['Set2', 'pastel', 'deep', 'colorblind', 'husl', 'Paired'],
            index=0,
            help="Change the color scheme for visualizations"
        )

# ----
# Apply filters
# ----
try:
    filtered_df = df[
        (df['age'].between(age_range[0], age_range[1])) &
        (df['balance'].between(balance_range[0], balance_range[1])) &
        (df['duration'].between(duration_range[0], duration_range[1])) &
        (df['job'].isin(job_filter)) &
        (df['marital'].isin(marital_filter)) &
        (df['education'].isin(education_filter)) &
        (df['loan'].isin(loan_filter))
    ].copy()
    
    if sample_size < len(filtered_df):
        filtered_df = filtered_df.sample(sample_size, random_state=42)
        
except Exception as e:
    st.error(f"Error applying filters: {str(e)}")
    filtered_df = df.copy()

# ----
# Main content
# ----
st.title("Bank Customer Analytics Dashboard")
st.markdown(f"""
    Interactive dashboard for comprehensive analysis of bank customer data.
    **Currently showing {len(filtered_df):,} customers** out of {len(df):,} total.
    Use the filters in the sidebar to customise your view.
""")

# ----
# Key Performance Indicators (using native st.metric, no custom CSS)
# ----
if len(filtered_df) > 0:
    st.subheader("Key Performance Indicators")
    cols = st.columns(4)
    
    total_customers = len(filtered_df)
    pct_of_total = (total_customers / len(df)) * 100
    cols[0].metric("Total Customers", f"{total_customers:,}", f"{pct_of_total:.1f}% of total")
    
    avg_balance = filtered_df['balance'].mean()
    median_balance = filtered_df['balance'].median()
    cols[1].metric("Average Balance", f"${avg_balance:,.0f}", f"Median: ${median_balance:,.0f}")
    
    loan_rate = (len(filtered_df[filtered_df['loan'] == 'yes']) / len(filtered_df)) * 100
    total_loan_rate = (len(df[df['loan'] == 'yes']) / len(df)) * 100
    rate_diff = loan_rate - total_loan_rate
    cols[2].metric("Loan Take-up Rate", f"{loan_rate:.1f}%", f"{rate_diff:+.1f}% vs overall")
    
    avg_duration = filtered_df['duration'].mean()
    overall_duration = df['duration'].mean()
    duration_diff = avg_duration - overall_duration
    cols[3].metric("Avg Call Duration", f"{avg_duration:.0f} sec", f"{duration_diff:+.0f} sec vs overall")
else:
    st.warning("No data matches the current filters. Please adjust your filter settings.")

# ----
# Export section (sidebar)
# ----
with st.sidebar.expander("Data Export", expanded=False):
    st.markdown("**Export Filtered Data**")
    
    export_format = st.radio(
        "Select Export Format:",
        options=['CSV', 'Excel', 'JSON'],
        index=0,
        horizontal=True
    )
    
    if len(filtered_df) > 0:
        if export_format == 'CSV':
            data = filtered_df.to_csv(index=False).encode('utf-8')
            mime = 'text/csv'
            ext = 'csv'
        elif export_format == 'Excel':
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='CustomerData')
            data = output.getvalue()
            mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            ext = 'xlsx'
        else:
            data = filtered_df.to_json(orient='records', indent=2).encode('utf-8')
            mime = 'application/json'
            ext = 'json'
        
        st.download_button(
            label=f"Download as {export_format}",
            data=data,
            file_name=f'filtered_bank_customers_{datetime.now().strftime("%Y%m%d_%H%M")}.{ext}',
            mime=mime,
            help=f"Download the currently filtered data as {export_format}"
        )
    else:
        st.info("No data to export. Adjust filters to include data.")

# ----
# Tabs – no emojis
# ----
if len(filtered_df) > 0:
    # Use the selected color palette throughout
    plotly_palette = getattr(px.colors.qualitative, color_palette)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Demographics", 
        "Financial Analysis", 
        "Engagement",
        "Trends",
        "Advanced Analytics"
    ])

    # 
    # Tab 1: Demographics
    # 
    with tab1:
        st.header("Customer Demographics Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Job Distribution")
            job_dist = filtered_df['job'].value_counts().reset_index()
            job_dist.columns = ['Job', 'Count']
            
            fig1 = px.bar(
                job_dist, 
                y='Job', 
                x='Count', 
                orientation='h',
                color='Job',
                text='Count',
                title='Customer Distribution by Job Type',
                color_discrete_sequence=plotly_palette
            )
            fig1.update_layout(showlegend=False, height=400)
            fig1.update_traces(textposition='outside')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Age Distribution")
            fig2 = px.histogram(
                filtered_df,
                x='age',
                nbins=20,
                title='Age Distribution',
                color_discrete_sequence=[plotly_palette[0]]
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Demographic Breakdown")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Marital Status**")
            marital_dist = filtered_df['marital'].value_counts()
            fig3 = px.pie(
                values=marital_dist.values,
                names=marital_dist.index,
                title='Marital Status Distribution',
                hole=0.3,
                color_discrete_sequence=plotly_palette
            )
            fig3.update_layout(height=300)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.markdown("**Education Level**")
            education_dist = filtered_df['education'].value_counts()
            fig4 = px.pie(
                values=education_dist.values,
                names=education_dist.index,
                title='Education Distribution',
                hole=0.3,
                color_discrete_sequence=plotly_palette
            )
            fig4.update_layout(height=300)
            st.plotly_chart(fig4, use_container_width=True)
        
        with col3:
            st.markdown("**Age Groups**")
            age_group_dist = filtered_df['age_group'].value_counts()
            fig5 = px.pie(
                values=age_group_dist.values,
                names=age_group_dist.index,
                title='Age Groups Distribution',
                hole=0.3,
                color_discrete_sequence=plotly_palette
            )
            fig5.update_layout(height=300)
            st.plotly_chart(fig5, use_container_width=True)

    # 
    # Tab 2: Financial Analysis
    # 
    with tab2:
        st.header("Financial Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Balance Distribution")
            fig6 = px.histogram(
                filtered_df,
                x='balance',
                nbins=30,
                title='Account Balance Distribution',
                marginal='box',
                color_discrete_sequence=[plotly_palette[0]]
            )
            st.plotly_chart(fig6, use_container_width=True)
        
        with col2:
            st.subheader("Balance by Job Type")
            fig7 = px.box(
                filtered_df,
                x='job',
                y='balance',
                title='Balance Distribution by Job',
                points='outliers' if show_outliers else False,
                color_discrete_sequence=plotly_palette
            )
            fig7.update_xaxes(tickangle=45)
            st.plotly_chart(fig7, use_container_width=True)
        
        st.subheader("Balance Categories Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            balance_cat_dist = filtered_df['balance_category'].value_counts()
            fig8 = px.bar(
                x=balance_cat_dist.index,
                y=balance_cat_dist.values,
                title='Distribution by Balance Category',
                text=balance_cat_dist.values,
                color_discrete_sequence=plotly_palette
            )
            fig8.update_traces(textposition='outside')
            st.plotly_chart(fig8, use_container_width=True)
        
        with col2:
            fig9 = px.box(
                filtered_df,
                x='loan',
                y='balance',
                title='Balance Distribution by Loan Status',
                points='outliers' if show_outliers else False,
                color_discrete_sequence=plotly_palette
            )
            st.plotly_chart(fig9, use_container_width=True)
        
        st.subheader("Financial Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Balance Statistics**")
            balance_stats = filtered_df['balance'].describe()
            st.dataframe(balance_stats.to_frame('Balance ($)').style.format('{:,.2f}'))
        
        with col2:
            st.markdown("**Balance by Demographics**")
            demographic_choice = st.selectbox(
                "Group by:",
                options=['job', 'education', 'marital', 'age_group'],
                key='financial_demo'
            )
            balance_by_demo = filtered_df.groupby(demographic_choice)['balance'].agg(['mean', 'median', 'std', 'count'])
            st.dataframe(balance_by_demo.style.format({'mean': '{:,.0f}', 'median': '{:,.0f}', 'std': '{:,.0f}'}))

    # 
    # Tab 3: Engagement
    # 
    with tab3:
        st.header("Customer Engagement Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Call Duration Distribution")
            fig10 = px.histogram(
                filtered_df,
                x='duration',
                nbins=25,
                title='Call Duration Distribution',
                marginal='box',
                color_discrete_sequence=[plotly_palette[0]]
            )
            st.plotly_chart(fig10, use_container_width=True)
        
        with col2:
            st.subheader("Duration by Loan Status")
            fig11 = px.box(
                filtered_df,
                x='loan',
                y='duration',
                title='Call Duration by Loan Status',
                points='outliers' if show_outliers else False,
                color_discrete_sequence=plotly_palette
            )
            st.plotly_chart(fig11, use_container_width=True)
        
        st.subheader("Call Duration Categories")
        col1, col2 = st.columns(2)
        
        with col1:
            duration_cat_dist = filtered_df['duration_category'].value_counts()
            fig12 = px.pie(
                values=duration_cat_dist.values,
                names=duration_cat_dist.index,
                title='Call Duration Categories',
                hole=0.3,
                color_discrete_sequence=plotly_palette
            )
            st.plotly_chart(fig12, use_container_width=True)
        
        with col2:
            loan_dist = filtered_df['loan'].value_counts()
            fig13 = px.pie(
                values=loan_dist.values,
                names=loan_dist.index,
                title='Loan Status Distribution',
                hole=0.3,
                color_discrete_sequence=plotly_palette
            )
            st.plotly_chart(fig13, use_container_width=True)
        
        st.subheader("Engagement Metrics Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_dur = filtered_df['duration'].mean()
            st.metric("Average Call Duration", f"{avg_dur:.0f} seconds")
            
        with col2:
            engagement_rate = len(filtered_df[filtered_df['duration'] > 180]) / len(filtered_df) * 100
            st.metric("High Engagement Rate", f"{engagement_rate:.1f}%", 
                     help="Percentage of calls longer than 3 minutes")
        
        with col3:
            quick_call_rate = len(filtered_df[filtered_df['duration'] < 60]) / len(filtered_df) * 100
            st.metric("Quick Call Rate", f"{quick_call_rate:.1f}%",
                     help="Percentage of calls shorter than 1 minute")

    # 
    # Tab 4: Trends
    # 
    with tab4:
        st.header("Time Trends Analysis")
        
        st.subheader("Monthly Customer Distribution")
        monthly_dist = filtered_df['month_name'].value_counts().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ])
        
        fig14 = px.line(
            x=monthly_dist.index,
            y=monthly_dist.values,
            title='Customer Count by Month',
            markers=True,
            color_discrete_sequence=[plotly_palette[0]]
        )
        fig14.update_xaxes(tickangle=45)
        st.plotly_chart(fig14, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Quarterly Balance Trends")
            quarterly_balance = filtered_df.groupby('quarter')['balance'].mean().reset_index()
            fig15 = px.bar(
                quarterly_balance,
                x='quarter',
                y='balance',
                title='Average Balance by Quarter',
                color_discrete_sequence=plotly_palette
            )
            st.plotly_chart(fig15, use_container_width=True)
        
        with col2:
            st.subheader("Quarterly Duration Trends") 
            quarterly_duration = filtered_df.groupby('quarter')['duration'].mean().reset_index()
            fig16 = px.bar(
                quarterly_duration,
                x='quarter',
                y='duration',
                title='Average Call Duration by Quarter',
                color_discrete_sequence=plotly_palette
            )
            st.plotly_chart(fig16, use_container_width=True)

    # 
    # Tab 5: Advanced Analytics
    # 
    with tab5:
        st.header("Advanced Analytics")
        
        st.subheader("Correlation Matrix")
        numeric_cols = ['age', 'balance', 'duration']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig17 = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect='auto',
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1,
            title='Correlation Matrix of Numeric Features'
        )
        st.plotly_chart(fig17, use_container_width=True)
        
        st.subheader("Feature Relationships")
        col1, col2 = st.columns(2)
        
        with col1:
            fig18 = px.scatter(
                filtered_df,
                x='age',
                y='balance',
                color='loan',
                title='Age vs Balance (colored by Loan Status)',
                opacity=0.6,
                color_discrete_sequence=plotly_palette
            )
            st.plotly_chart(fig18, use_container_width=True)
        
        with col2:
            fig19 = px.scatter(
                filtered_df,
                x='duration',
                y='balance',
                color='job',
                title='Call Duration vs Balance (colored by Job)',
                opacity=0.6,
                color_discrete_sequence=plotly_palette
            )
            st.plotly_chart(fig19, use_container_width=True)
        
        st.subheader("Statistical Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Key Statistics**")
            st.write(f"- Total customers analyzed: {len(filtered_df):,}")
            st.write(f"- Average age: {filtered_df['age'].mean():.1f} years")
            st.write(f"- Balance std deviation: ${filtered_df['balance'].std():,.0f}")
            st.write(f"- Duration range: {filtered_df['duration'].min()} – {filtered_df['duration'].max()} seconds")
            age_balance_corr = filtered_df['age'].corr(filtered_df['balance'])
            st.write(f"- Age–Balance correlation: {age_balance_corr:.3f}")
            
        with col2:
            st.markdown("**Data Quality Metrics**")
            missing_data = filtered_df.isnull().sum()
            if missing_data.sum() > 0:
                st.write("Missing data by column:")
                for col, missing in missing_data.items():
                    if missing > 0:
                        st.write(f"- {col}: {missing} ({missing/len(filtered_df)*100:.1f}%)")
            else:
                st.write("No missing data in the current selection.")

else:
    st.info("Adjust the filters to see analytics and visualizations.")
