import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import random
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Amazon Review Analyzer", page_icon="📊", layout="wide")

st.title("🛍️ Amazon Product Review Intelligence")
st.markdown("### Analyze sentiment patterns in Amazon product reviews")

# Generate realistic Amazon review data
@st.cache_data
def generate_amazon_reviews():
    products = [
        "iPhone 15 Pro Max",
        "Samsung Galaxy S24", 
        "MacBook Air M2",
        "Sony WH-1000XM5 Headphones",
        "Kindle Paperwhite",
        "Echo Dot (5th Gen)",
        "Fire TV Stick 4K",
        "Ring Video Doorbell"
    ]
    
    review_templates = {
        "positive": [
            "Amazing product! {feature} is incredible. {benefit}",
            "Best purchase this year. {feature} works flawlessly.",
            "5 stars! {feature} exceeded expectations. {benefit}",
            "Love it! {feature} is game-changing.",
            "Worth every penny. {feature} and {benefit} are top notch."
        ],
        "negative": [
            "Disappointed. {issue} is a major problem.",
            "Not worth the money. {issue} needs improvement.",
            "2 stars only. {issue} and {problem} ruined it.",
            "Expected better. {issue}",
            "Returning this. {problem} is unacceptable."
        ],
        "neutral": [
            "Decent product. {feature} is okay for the price.",
            "Average. {feature} works but {issue} could be better.",
            "It's fine. Does the job but nothing special.",
            "Mixed feelings. {benefit} but {problem}",
            "Standard product. Meets basic expectations."
        ]
    }
    
    features = {
        "iPhone": ["Camera", "Battery life", "Display", "Performance", "Build quality"],
        "Samsung": ["Screen", "Camera", "Battery", "Speed", "Design"],
        "MacBook": ["Performance", "Battery", "Screen", "Keyboard", "Trackpad"],
        "Sony": ["Sound quality", "Noise cancellation", "Comfort", "Battery", "Connectivity"],
        "Kindle": ["Screen", "Battery", "Light weight", "Storage", "Reading experience"],
        "Echo": ["Voice recognition", "Sound", "Setup", "Alexa", "Smart home control"],
        "Fire TV": ["Streaming quality", "Remote", "Interface", "Speed", "Apps"],
        "Ring": ["Video quality", "Installation", "Motion detection", "App", "Notifications"]
    }
    
    reviews = []
    
    for product in products:
        product_category = product.split()[0]
        product_features = features.get(product_category, ["Quality", "Price", "Value"])
        
        # Generate reviews over last 6 months
        for i in range(30):
            sentiment = random.choices(["positive", "negative", "neutral"], weights=[0.6, 0.2, 0.2])[0]
            template = random.choice(review_templates[sentiment])
            
            feature = random.choice(product_features)
            benefit = random.choice(["saves time", "great value", "premium feel", "easy to use"])
            issue = random.choice(["glitchy software", "poor battery", "expensive", "complicated"])
            problem = random.choice(["crashes often", "overheats", "too bulky", "poor support"])
            
            review = template.format(
                feature=feature,
                benefit=benefit,
                issue=issue,
                problem=problem
            )
            
            if sentiment == "positive":
                rating = random.randint(4, 5)
            elif sentiment == "negative":
                rating = random.randint(1, 2)
            else:
                rating = 3
            
            # Random date in last 180 days
            random_days = random.randint(0, 180)
            review_date = datetime.now() - timedelta(days=random_days)
            
            reviews.append({
                "product": product,
                "review": review,
                "rating": rating,
                "sentiment": sentiment,
                "date": review_date.strftime("%Y-%m-%d")
            })
    
    return pd.DataFrame(reviews)

# Load data
df = generate_amazon_reviews()

# Sidebar filters
st.sidebar.header("🔍 Filter Reviews")
selected_products = st.sidebar.multiselect(
    "Select Products",
    options=df['product'].unique(),
    default=list(df['product'].unique()[:3])
)

min_rating = st.sidebar.slider("Minimum Rating", 1, 5, 1)

# Filter data
filtered_df = df[df['product'].isin(selected_products) & (df['rating'] >= min_rating)]

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Reviews", len(filtered_df))

with col2:
    avg_rating = filtered_df['rating'].mean()
    st.metric("Average Rating", f"{avg_rating:.2f} ⭐")

with col3:
    pos_pct = (filtered_df['sentiment'] == 'positive').mean() * 100
    st.metric("Positive Reviews", f"{pos_pct:.1f}%")

with col4:
    neg_pct = (filtered_df['sentiment'] == 'negative').mean() * 100
    st.metric("Negative Reviews", f"{neg_pct:.1f}%")

# Charts
col1, col2 = st.columns(2)

with col1:
    fig_ratings = px.histogram(
        filtered_df, 
        x='rating', 
        color='product',
        title="Rating Distribution by Product",
        nbins=5,
        barmode='group'
    )
    st.plotly_chart(fig_ratings, use_container_width=True)

with col2:
    sentiment_counts = filtered_df['sentiment'].value_counts()
    fig_sentiment = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=0.4,
        marker_colors=['#2ecc71', '#e74c3c', '#f39c12']
    )])
    fig_sentiment.update_layout(title="Sentiment Breakdown")
    st.plotly_chart(fig_sentiment, use_container_width=True)

# Product selector for deep dive
st.header("🔬 Deep Review Analysis")
selected_product = st.selectbox("Select product for detailed analysis", filtered_df['product'].unique())
product_df = filtered_df[filtered_df['product'] == selected_product]

col1, col2 = st.columns(2)

with col1:
    all_words = ' '.join(product_df['review']).lower().split()
    stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    words = [w for w in all_words if w not in stop_words and len(w) > 3]
    common_words = Counter(words).most_common(10)
    
    if common_words:
        words_df = pd.DataFrame(common_words, columns=['word', 'count'])
        fig_words = px.bar(words_df, x='word', y='count', title=f"Top Keywords in {selected_product} Reviews")
        st.plotly_chart(fig_words, use_container_width=True)

with col2:
    product_df['date'] = pd.to_datetime(product_df['date'])
    
    # Sentiment over time (by week)
    product_df['week'] = product_df['date'].dt.to_period('W').astype(str)
    time_sentiment = product_df.groupby(['week', 'sentiment']).size().reset_index(name='count')
    
    fig_time = px.line(
        time_sentiment, 
        x='week', 
        y='count', 
        color='sentiment',
        title="Sentiment Trends Over Time",
        color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
    )
    st.plotly_chart(fig_time, use_container_width=True)

# Sample reviews
st.header("📝 Sample Reviews")

tab1, tab2, tab3 = st.tabs(["Positive 😊", "Negative 😠", "Neutral 😐"])

with tab1:
    positive_reviews = product_df[product_df['sentiment'] == 'positive'].head(5)
    for _, row in positive_reviews.iterrows():
        st.markdown(f"⭐ {row['rating']}/5 - {row['review']}")
        st.caption(f"📅 {row['date']}")
        st.divider()

with tab2:
    negative_reviews = product_df[product_df['sentiment'] == 'negative'].head(5)
    for _, row in negative_reviews.iterrows():
        st.markdown(f"⭐ {row['rating']}/5 - {row['review']}")
        st.caption(f"📅 {row['date']}")
        st.divider()

with tab3:
    neutral_reviews = product_df[product_df['sentiment'] == 'neutral'].head(5)
    for _, row in neutral_reviews.iterrows():
        st.markdown(f"⭐ {row['rating']}/5 - {row['review']}")
        st.caption(f"📅 {row['date']}")
        st.divider()

# Footer
st.markdown("---")
st.markdown("💡 **Built with Streamlit • Data simulates Amazon product reviews • For demonstration purposes**")
