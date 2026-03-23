import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from io import BytesIO
from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile
from datetime import datetime
st.sidebar.markdown("<h1 style='text-align: center; color: #38bdf8;'>💠 NEXUS AI</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")
def get_session_duration():
    if "login_time" in st.session_state and st.session_state.login_time is not None:
        duration = datetime.now() - st.session_state.login_time
        # Removes microseconds for a clean HH:MM:SS format
        return str(duration).split(".")[0] 
    return "0:00:00"

st.set_page_config(page_title="AI Business Analytics Platform", layout="wide")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.main {background-color:#0f172a;}
h1,h2,h3,h4 {color:white;}
section[data-testid="stSidebar"] {background-color:#020617;}
.stButton>button {background-color:#38bdf8;color:white;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "auth" not in st.session_state: st.session_state.auth=False
if "users" not in st.session_state: st.session_state.users={"admin":{"password":"admin123","role":"Admin","blocked":False}}
if "logs" not in st.session_state: st.session_state.logs=[] 
if "data" not in st.session_state: st.session_state.data=None
if "page" not in st.session_state: st.session_state.page="login"

# ---------------- REGISTER ----------------
def register():
    st.title("📝 Create New Account")
    st.subheader("Join the Analytics Platform")
    
    new_u = st.text_input("Choose Username")
    new_p = st.text_input("Choose Password", type="password")
    
    # --- PASSWORD STRENGTH METER ---
    if new_p:
        # Simple Logic: Score based on length and variety
        strength = 0
        if len(new_p) >= 6: strength += 0.3
        if len(new_p) >= 10: strength += 0.3
        if any(char.isdigit() for char in new_p): strength += 0.2
        if any(not char.isalnum() for char in new_p): strength += 0.2
        
        # Determine color and label
        if strength <= 0.3:
            label, color = "Weak", "red"
        elif strength <= 0.6:
            label, color = "Fair", "orange"
        else:
            label, color = "Strong", "green"
            
        st.write(f"Strength: :{color}[**{label}**]")
        st.progress(strength)
    # -------------------------------

    confirm_p = st.text_input("Confirm Password", type="password")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Register Now", use_container_width=True):
            if new_u == "" or new_p == "":
                st.warning("⚠️ Fields cannot be empty")
            elif new_u in st.session_state.users:
                st.error("❌ Username already exists")
            elif new_p != confirm_p:
                st.error("❌ Passwords do not match")
            elif len(new_p) < 6:
                st.error("❌ Password must be at least 6 characters")
            else:
                st.session_state.users[new_u] = {
                    "password": new_p, 
                    "role": "User", 
                    "blocked": False
                }
                st.success("✅ Registration Successful! Please Login.")
                
    with col2:
        if st.button("Back to Login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()
# ---------------- LOGIN ----------------
# ---------------- LOGIN ----------------
def login():
    st.title("🚀 AI Business Analytics Platform")
    st.subheader("Login to your account")
    
    u = st.text_input("Username", placeholder="Enter your username")
    p = st.text_input("Password", type="password", placeholder="Enter your password")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Login", use_container_width=True):
            users = st.session_state.users
            if u in users:
                if users[u]["blocked"]: 
                    st.error("🚫 Access Denied: This account is blocked.")
                elif users[u]["password"] == p:
                    # Initialize Session
                    st.session_state.auth = True
                    st.session_state.username = u
                    st.session_state.role = users[u]["role"]
                    st.session_state.login_time = datetime.now() # START TRIGGER
                    
                    # Log the event
                    st.session_state.logs.append({
                        "User": u, 
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                        "Event": "Login Success"
                    })
                    st.rerun()
                else: 
                    st.error("❌ Wrong password")
            else: 
                st.error("❓ User not found")
                
    with col2:
        if st.button("Create Account", use_container_width=True):
            st.session_state.page = "register"
            st.rerun()
# ---------------- DATA LOAD ----------------
def load_data(file):
    try:
        file_type = file.name.split('.')[-1].lower()

        # ---- CSV ----
        if file_type == "csv":
            df = pd.read_csv(file)

        # ---- Excel ----
        elif file_type in ["xlsx", "xls"]:
            df = pd.read_excel(file)

        # ---- JSON ----
        elif file_type == "json":
            df = pd.read_json(file)

        # ---- TXT (try CSV format) ----
        elif file_type == "txt":
            df = pd.read_csv(file, delimiter=None, engine='python')

        # ---- Unsupported → Auto Try ----
        else:
            try:
                df = pd.read_csv(file)
            except:
                st.error("❌ Unsupported file format")
                return None

        # ---- Auto Column Handling ----
        df.columns = df.columns.str.strip()

        # ---- Smart Column Detection ----
        if "Date" not in df.columns:
            possible_date = [col for col in df.columns if "date" in col.lower()]
            if possible_date:
                df.rename(columns={possible_date[0]: "Date"}, inplace=True)
            else:
                st.error("Dataset must have a Date column")
                return None

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # ---- Default Columns (if missing) ----
        if "Revenue" not in df.columns:
            df["Revenue"] = np.random.randint(100, 1000, len(df))

        if "Units_Sold" not in df.columns:
            df["Units_Sold"] = np.random.randint(1, 50, len(df))

        if "Product" not in df.columns:
            df["Product"] = "Unknown"

        if "Region" not in df.columns:
            df["Region"] = "Unknown"

        return df

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

# ---------------- EXPORT EXCEL ----------------
def export_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        summary_data = {
            "Metric": ["Total Revenue", "Total Orders", "Units Sold", "Avg Order Value"],
            "Value": [df['Revenue'].sum(), len(df), df['Units_Sold'].sum(), df['Revenue'].mean()]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
        top10 = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False).head(10).reset_index()
        top10.to_excel(writer, sheet_name="Top Products", index=False)
        regions = df.groupby("Region")["Revenue"].sum().reset_index()
        regions.to_excel(writer, sheet_name="Regional Analysis", index=False)
        df.to_excel(writer, sheet_name="Full Dataset", index=False)
        workbook = writer.book
        for sheet in writer.sheets.values():
            sheet.set_column('A:Z', 15)
    return buffer.getvalue()

# ---------------- EXPORT PDF ----------------
def export_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(56, 189, 248)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 12, "AI Sales Dashboard Report", ln=True, align="C", fill=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.set_fill_color(200,200,200)
    pdf.cell(60,10,"Total Revenue",border=1,fill=True)
    pdf.cell(60,10,"Total Orders",border=1,fill=True)
    pdf.cell(60,10,"Units Sold",border=1,fill=True)
    pdf.ln()
    pdf.set_font("Arial", "", 12)
    pdf.cell(60,10,f"${df['Revenue'].sum():,.2f}",border=1)
    pdf.cell(60,10,f"{len(df)}",border=1)
    pdf.cell(60,10,f"{df['Units_Sold'].sum()}",border=1)
    pdf.ln(15)
    
    pdf.set_font("Arial","B",12)
    pdf.set_fill_color(200,200,200)
    pdf.cell(100,10,"Top 10 Products",border=1,fill=True)
    pdf.cell(80,10,"Revenue",border=1,fill=True)
    pdf.ln()
    pdf.set_font("Arial","",12)
    top10=df.groupby("Product")["Revenue"].sum().sort_values(ascending=False).head(10)
    for product,revenue in top10.items():
        pdf.cell(100,10,product,border=1)
        pdf.cell(80,10,f"${revenue:,.2f}",border=1)
        pdf.ln()
    pdf.ln(10)
    
    pdf.set_font("Arial","B",12)
    pdf.set_fill_color(200,200,200)
    pdf.cell(100,10,"Region",border=1,fill=True)
    pdf.cell(80,10,"Revenue",border=1,fill=True)
    pdf.ln()
    region_total=df.groupby("Region")["Revenue"].sum()
    for region,revenue in region_total.items():
        pdf.cell(100,10,region,border=1)
        pdf.cell(80,10,f"${revenue:,.2f}",border=1)
        pdf.ln()
    pdf.ln(10)
    
    df_plot = df.groupby("Date").agg({"Revenue":"sum","Units_Sold":"sum"}).reset_index()
    df_plot["Profit"] = df_plot["Revenue"]*0.3
    mini_charts = [("Revenue","Revenue Trend"),("Profit","Profit Trend"),("Units_Sold","Units Sold Trend")]
    for col,title in mini_charts:
        pdf.set_font("Arial","B",12)
        pdf.cell(0,6,title,ln=True)
        plt.figure(figsize=(4,1))
        plt.plot(df_plot["Date"], df_plot[col], color="blue")
        plt.axis("off")
        plt.tight_layout()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name, format="PNG", dpi=150)
            plt.close()
            pdf.image(tmpfile.name, x=30, w=150)
        pdf.ln(10)
    return pdf.output(dest="S").encode("latin-1")

# ---------------- DASHBOARD ----------------
def dashboard():
    st.sidebar.title("📊 Analytics Platform")
    st.sidebar.write("Welcome", st.session_state.username)
    if st.sidebar.button("Logout"):
        st.session_state.auth=False; st.rerun()

    page=st.sidebar.selectbox("Navigation",["Upload Data","Data Health Check","Dataset Preview","Inventory Management","KPI Dashboard","Analytics Charts","Forecast","Segmentation","Anomaly Detection","AI Insights","AI Assistant","Export","Admin Panel"])
    df = st.session_state.data

    if page=="Upload Data":
        st.header("📥 Data Acquisition")
        t1, t2 = st.tabs(["📁 File Upload", "📝 Manual Data Entry"])
        with t1:
            file = st.file_uploader(
    "Upload ANY dataset file",
    type=["csv", "xlsx", "xls", "json", "txt"]
)
            if file:
                df=load_data(file)
                if df is not None: st.session_state.data=df; st.success("Dataset Loaded")
        with t2:
            st.write("Add individual records manually:")
            with st.form("manual_form"):
                col1, col2, col3 = st.columns(3)
                m_date = col1.date_input("Date", datetime.now())
                m_prod = col2.text_input("Product Name")
                m_reg = col3.selectbox("Region", ["North", "South", "East", "West", "International"])
                col4, col5 = st.columns(2)
                m_rev = col4.number_input("Revenue", min_value=0.0)
                m_units = col5.number_input("Units Sold", min_value=0)
                if st.form_submit_button("Add Record"):
                    new_data = pd.DataFrame({"Date":[pd.to_datetime(m_date)], "Product":[m_prod], "Region":[m_reg], "Revenue":[m_rev], "Units_Sold":[m_units]})
                    st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True) if st.session_state.data is not None else new_data
                    st.success("Record Added!")
                    st.rerun()
            if st.button("Clear All Data"):
                st.session_state.data = None
                st.rerun()

    elif page=="Data Health Check":
        if df is not None:
            st.header("🩺 Data Health & Integrity Report")
            c1, c2, c3 = st.columns(3)
            c1.metric("Missing Values", df.isna().sum().sum())
            c2.metric("Duplicate Rows", df.duplicated().sum())
            c3.metric("Total Rows", len(df))
            st.table(pd.DataFrame({"Column": df.columns,"Type": df.dtypes.astype(str),"Nulls": df.isnull().sum().values}))
        else: st.warning("Upload dataset first")

    elif page=="Inventory Management":
        st.header("📦 Inventory Tracking & COGS")
        if df is not None:
            inv_df = df.groupby("Product").agg({"Units_Sold":"sum"}).reset_index()
            inv_df["Stock_Level"] = np.random.randint(10, 100, size=len(inv_df))
            inv_df["Unit_Cost"] = np.random.uniform(5.0, 50.0, size=len(inv_df))
            inv_df["Total_Value"] = inv_df["Stock_Level"] * inv_df["Unit_Cost"]
            c1, c2 = st.columns(2)
            c1.metric("Total Stock Items", inv_df["Stock_Level"].sum())
            c2.metric("Total COGS Value", f"${inv_df['Total_Value'].sum():,.2f}")
            styled_inv = inv_df.style.highlight_between(left=0, right=15, subset=['Stock_Level'], color='#8B0000').format({"Unit_Cost": "${:.2f}", "Total_Value": "${:.2f}"})
            st.dataframe(styled_inv, use_container_width=True)
            st.plotly_chart(px.bar(inv_df, x="Product", y="Stock_Level", title="Stock Levels by Product", color="Stock_Level"), use_container_width=True)
        else: st.warning("Upload dataset first")

    elif page=="Dataset Preview":
        if df is not None: st.dataframe(df)

    elif page=="KPI Dashboard":
        if df is not None:
            c1,c2,c3,c4=st.columns(4)
            c1.markdown(f"<div style='background:#1e293b;padding:25px;border-radius:15px;text-align:center;'><h4 style='color:#94a3b8'>Total Revenue</h4><h1 style='color:#38bdf8'>${df['Revenue'].sum():,.2f}</h1></div>", unsafe_allow_html=True)
            c2.markdown(f"<div style='background:#1e293b;padding:25px;border-radius:15px;text-align:center;'><h4 style='color:#94a3b8'>Orders</h4><h1 style='color:#38bdf8'>{len(df)}</h1></div>", unsafe_allow_html=True)
            c3.markdown(f"<div style='background:#1e293b;padding:25px;border-radius:15px;text-align:center;'><h4 style='color:#94a3b8'>Avg Revenue</h4><h1 style='color:#38bdf8'>${df['Revenue'].mean():,.2f}</h1></div>", unsafe_allow_html=True)
            c4.markdown(f"<div style='background:#1e293b;padding:25px;border-radius:15px;text-align:center;'><h4 style='color:#94a3b8'>Units Sold</h4><h1 style='color:#38bdf8'>{df['Units_Sold'].sum()}</h1></div>", unsafe_allow_html=True)

    filtered_df=None
    if df is not None:
        st.sidebar.subheader("Filter Data")
        s_p=st.sidebar.multiselect("Products",list(df["Product"].unique()),list(df["Product"].unique()))
        s_r=st.sidebar.multiselect("Regions",list(df["Region"].unique()),list(df["Region"].unique()))
        filtered_df=df[(df["Product"].isin(s_p)) & (df["Region"].isin(s_r))]
    if page=="Analytics Charts" and filtered_df is not None:
        st.plotly_chart(px.line(filtered_df,x="Date",y="Revenue",title="1. Revenue Trend Line"),use_container_width=True)
        st.plotly_chart(px.bar(filtered_df,x="Product",y="Revenue",title="2. Revenue by Product"),use_container_width=True)
        st.plotly_chart(px.pie(filtered_df,names="Region",values="Revenue",title="3. Regional Revenue Share"),use_container_width=True)
        st.plotly_chart(px.area(filtered_df,x="Date",y="Units_Sold",title="4. Units Sold Over Time"),use_container_width=True)
        st.plotly_chart(px.box(filtered_df,x="Product",y="Revenue",title="5. Revenue Distribution"),use_container_width=True)
        st.plotly_chart(px.histogram(filtered_df,x="Revenue",title="6. Revenue Histogram"),use_container_width=True)
        st.plotly_chart(px.scatter(filtered_df,x="Units_Sold",y="Revenue",color="Product",title="7. Units Sold vs Revenue Scatter"),use_container_width=True)
        st.plotly_chart(px.violin(filtered_df, y="Revenue", x="Region", box=True, title="8. Regional Revenue Violin"),use_container_width=True)
        st.plotly_chart(px.funnel(filtered_df.groupby("Product")["Revenue"].sum().reset_index(), x='Revenue', y='Product', title="9. Sales Funnel"),use_container_width=True)
        st.plotly_chart(px.density_heatmap(filtered_df, x="Units_Sold", y="Revenue", title="10. Revenue Density Heatmap"),use_container_width=True)
        st.plotly_chart(px.sunburst(filtered_df, path=['Region', 'Product'], values='Revenue', title="11. Region-Product Hierarchy"),use_container_width=True)
        st.plotly_chart(px.strip(filtered_df, x="Revenue", y="Product", title="12. Sales Strip Plot"),use_container_width=True)
        st.plotly_chart(px.ecdf(filtered_df, x="Revenue", title="13. Cumulative Distribution"),use_container_width=True)
        st.plotly_chart(px.bar(filtered_df.groupby("Product")["Units_Sold"].sum().reset_index(), x="Product", y="Units_Sold", title="14. Units Sold by Product"),use_container_width=True)
        st.plotly_chart(px.line(filtered_df, x="Date", y="Units_Sold", color="Product", title="15. Product Unit Trends"),use_container_width=True)
        st.plotly_chart(px.scatter(filtered_df, x="Date", y="Revenue", size="Units_Sold", title="16. Revenue Bubble Chart"),use_container_width=True)
        st.plotly_chart(px.histogram(filtered_df, x="Units_Sold", title="17. Units Distribution"),use_container_width=True)
        st.plotly_chart(px.bar(filtered_df, x="Region", y="Units_Sold", color="Product", barmode="group", title="18. Regional Units by Product"),use_container_width=True)
        st.plotly_chart(px.pie(filtered_df, names="Product", values="Units_Sold", hole=.3, title="19. Units Sold Donut Chart"),use_container_width=True)
        st.plotly_chart(px.line(filtered_df.groupby("Date")["Revenue"].mean().reset_index(), x="Date", y="Revenue", title="20. Avg Daily Revenue"),use_container_width=True)

    elif page == "Forecast" and filtered_df is not None:
        st.header("📈 AI Strategy & Predictive Forecasting")
        
        # 1. Prepare Data
        df_prophet = filtered_df.groupby("Date")["Revenue"].sum().reset_index().rename(columns={"Date":"ds", "Revenue":"y"})
        
        if len(df_prophet) > 10:
            with st.spinner("Analyzing trends and seasonal patterns..."):
                # 2. Model Configuration
                model = Prophet(growth='linear', seasonality_mode='multiplicative', changepoint_prior_scale=0.05)
                model.add_country_holidays(country_name='IN')  # Auto-detect Indian Festivals
                model.fit(df_prophet)
                
                # 3. Forecast Generation
                days_to_predict = st.slider("Forecast Horizon (Days)", 7, 90, 30)
                future = model.make_future_dataframe(periods=days_to_predict)
                forecast = model.predict(future)
                
                # 4. Accuracy Validation (Backtesting)
                with st.expander("🎯 Model Reliability & Accuracy"):
                    try:
                        from prophet.diagnostics import cross_validation, performance_metrics
                        # Backtest using a 30-day initial window
                        df_cv = cross_validation(model, initial='30 days', period='7 days', horizon='14 days')
                        df_p = performance_metrics(df_cv)
                        avg_mape = df_p['mape'].mean() * 100
                        accuracy = 100 - avg_mape
                        
                        m1, m2 = st.columns(2)
                        m1.metric("Prediction Accuracy", f"{accuracy:.1f}%")
                        m2.metric("Avg. Error (MAPE)", f"{avg_mape:.1f}%")
                        st.progress(max(0, min(accuracy/100, 1.0)))
                    except:
                        st.info("More historical data is needed to calculate precise accuracy scores.")

                # 5. Visualizations
                c1, c2 = st.columns([3, 1])
                with c1:
                    fig = px.line(forecast, x='ds', y='yhat', title="Revenue Projection")
                    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], line=dict(width=0), name='Upper Confidence', showlegend=False)
                    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', line=dict(width=0), name='Lower Confidence', showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with c2:
                    st.subheader("Key Insights")
                    expected_rev = forecast.iloc[-1]['yhat']
                    st.metric("Target Revenue", f"${expected_rev:,.2f}")
                    
                    # 6. Export Forecast Data
                    csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Forecast CSV", data=csv, file_name="sales_forecast.csv", mime="text/csv", use_container_width=True)

                # 7. Seasonal Breakdown
                with st.expander("📅 Seasonal Trends (Weekly & Yearly)"):
                    fig_comp = model.plot_components(forecast)
                    st.write(fig_comp)
        else:
            st.warning("Please upload more data (at least 10 unique dates) to enable AI Forecasting.")
    elif page=="Segmentation" and filtered_df is not None:
         seg_df = filtered_df.copy(); seg_df["Cluster"]=KMeans(n_clusters=3).fit_predict(seg_df[["Revenue","Units_Sold"]])
         st.plotly_chart(px.scatter(seg_df,x="Revenue",y="Units_Sold",color="Cluster"),use_container_width=True)

    elif page=="Anomaly Detection" and filtered_df is not None:
        anomaly_df = filtered_df.copy(); anomaly_df["Anomaly"]=IsolationForest(contamination=0.05).fit_predict(anomaly_df[["Revenue","Units_Sold"]])
        st.plotly_chart(px.scatter(anomaly_df,x="Revenue",y="Units_Sold",color="Anomaly"),use_container_width=True)

    elif page == "AI Insights" and filtered_df is not None:
        st.header("🧠 Predictive Intelligence & Drivers")
        
        # --- 1. FEATURE IMPORTANCE (What drives Revenue?) ---
        # We'll use a simple correlation matrix or a random forest regressor 
        # to show which factors impact revenue the most.
        from sklearn.ensemble import RandomForestRegressor
        
        st.subheader("What's Driving Your Revenue?")
        
        # Prepare data for a quick model
        df_ml = filtered_df.copy()
        # Simple encoding for demo
        df_ml['Region_Code'] = df_ml['Region'].astype('category').cat.codes
        df_ml['Product_Code'] = df_ml['Product'].astype('category').cat.codes
        
        X = df_ml[['Region_Code', 'Product_Code', 'Units_Sold']]
        y = df_ml['Revenue']
        
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        
        # Plotting Importance
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        fig_ext = px.bar(feat_importances, orientation='h', 
                         title="Impact Factor (Contribution to Revenue)",
                         labels={'value': 'Importance Score', 'index': 'Variable'})
        st.plotly_chart(fig_ext, use_container_width=True)

        # --- 2. THE "WHAT-IF" SIMULATOR ---
        st.divider()
        st.subheader("🎮 Strategy Simulator (What-If?)")
        st.write("Adjust the sliders to simulate how changes in volume or pricing might impact your bottom line.")
        
        col_s1, col_s2 = st.columns(2)
        price_mod = col_s1.slider("Simulated Price Change (%)", -50, 100, 0)
        vol_mod = col_s2.slider("Simulated Volume Change (%)", -50, 100, 0)
        
        current_rev = filtered_df['Revenue'].sum()
        # Simple elasticity simulation
        simulated_rev = current_rev * (1 + (price_mod/100)) * (1 + (vol_mod/100))
        diff = simulated_rev - current_rev
        
        st.metric("Simulated Total Revenue", f"${simulated_rev:,.2f}", 
                  delta=f"${diff:,.2f}", delta_color="normal")
    elif page=="AI Assistant" and filtered_df is not None:
        q=st.text_input("Ask about dataset")
        if q and "revenue" in q.lower(): st.write("Total Revenue:",filtered_df["Revenue"].sum())

    elif page=="Export" and filtered_df is not None:
        st.download_button("Download Excel Report",data=export_excel(filtered_df),file_name="AI_Business_Report.xlsx")
        st.download_button("Download PDF Report",data=export_pdf(filtered_df),file_name="AI_Business_Report.pdf")

    elif page=="Admin Panel":
        if st.session_state.role=="Admin":
            st.header("🛡️ Admin Command Center")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Users", len(st.session_state.users))
            c2.metric("System Logs", len(st.session_state.logs))
            c3.metric("System Status", "Healthy", delta="Online")
            st.divider()
            st.subheader("👥 User Account Control")
            for username, info in list(st.session_state.users.items()):
                row = st.columns([2, 1, 1, 1, 1, 1])
                row[0].write(username)
                row[1].write(info['role'])
                row[2].write("🔴 Blocked" if info['blocked'] else "🟢 Active")
                if username != "admin":
                    if row[3].button("Toggle Block", key=f"bl_{username}"):
                        st.session_state.users[username]['blocked'] = not info['blocked']
                        st.rerun()
                    if row[4].button("Reset PW", key=f"reset_{username}"):
                        st.session_state.users[username]['password'] = "1234"
                        st.info(f"PW for {username} reset to 1234")
                    if row[5].button("Delete", key=f"dl_{username}"):
                        del st.session_state.users[username]
                        st.rerun()
                else: row[3].write("Protected")
            if st.session_state.logs:
                st.subheader("System Logs")
                st.table(pd.DataFrame(st.session_state.logs))
        else: st.error("🚫 Access Denied.")

if st.session_state.auth:
    dashboard()
else:
    # This logic handles which "Logged Out" screen to show
    if st.session_state.page == "login":
        login()
    elif st.session_state.page == "register":
        register()