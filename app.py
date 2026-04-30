import streamlit as st
import pandas as pd
import pickle
import ast

st.set_page_config(
    page_title="Retail Segmentation App",
    page_icon="🛒",
    layout="wide"
)

km = pickle.load(open("kmeans_model.pkl", "rb"))
sc = pickle.load(open("scaler.pkl", "rb"))

cluster_names = pd.read_csv("cluster_names.csv")
cluster_summary = pd.read_csv("cluster_summary.csv")
marketing = pd.read_csv("customer_marketing_recommendations.csv")
category_summary = pd.read_csv("customer_category_summary.csv")
recommendations = pd.read_csv("customer_recommendations.csv")


try:
    product_segments = pd.read_csv("product_segments.csv")
    product_cluster_summary = pd.read_csv("product_cluster_summary.csv")
    has_products = True
except:
    has_products = False

customer_descriptions = {
    "VIP customers": {
        "emoji": "💎",
        "description": "High-value customers with high spending, large baskets and recent purchases.",
        "recommendation": "Keep them loyal: personal offers, premium service, loyalty bonuses."
    },
    "Regular customers": {
        "emoji": "🟢",
        "description": "Stable customers with regular purchases and medium spending.",
        "recommendation": "Increase average check with bundles, cross-sell and targeted promotions."
    },
    "At-risk customers": {
        "emoji": "⚠️",
        "description": "Customers who purchased before but have not bought recently.",
        "recommendation": "Use reactivation campaigns: discounts, reminders and personalized offers."
    },
    "Low value customers": {
        "emoji": "🔵",
        "description": "Customers with low total spending and smaller baskets.",
        "recommendation": "Use entry-level promotions and simple offers to increase engagement."
    },
    "Low-value customers": {
        "emoji": "🔵",
        "description": "Customers with low total spending and smaller baskets.",
        "recommendation": "Use entry-level promotions and simple offers to increase engagement."
    }
}

product_descriptions = {
    "Top revenue drivers": {
        "emoji": "🔥",
        "description": "Products that generate the highest revenue.",
        "recommendation": "Keep stock availability high and use them as strategic products."
    },
    "Core products": {
        "emoji": "⭐",
        "description": "Stable products with strong demand and good customer reach.",
        "recommendation": "Use them in bundles and regular promotions."
    },
    "Mid-performing products": {
        "emoji": "📊",
        "description": "Products with average performance.",
        "recommendation": "Test discounts, improve visibility and analyze category fit."
    },
    "Underperforming products": {
        "emoji": "📉",
        "description": "Products with the lowest revenue or demand.",
        "recommendation": "Review pricing, placement or consider reducing stock."
    },
    "High revenue products": {
        "emoji": "🔥",
        "description": "Products that generate the highest revenue.",
        "recommendation": "Keep stock availability high and use them as strategic products."
    },
    "Popular products": {
        "emoji": "⭐",
        "description": "Products with stable demand and broad customer reach.",
        "recommendation": "Use them in bundles and regular promotions."
    },
    "Low performing products": {
        "emoji": "📉",
        "description": "Products with the weakest performance.",
        "recommendation": "Review pricing, placement or consider reducing stock."
    }
}


# =========================
# Header
# =========================
st.title("🛒 Retail Customer & Product Segmentation")
st.write(
    "This app helps managers understand customer behavior and product performance "
    "using KMeans clustering."
)

tab1, tab2 ,tab3= st.tabs(["👤 Customer Segment Predictor", "📦 Product Segmentation", "🎯 Personalized Marketing Recommendation"])


with tab1:
    st.header("👤 Customer Segment Predictor")

    st.markdown(
        """
        Enter customer behavior metrics below.  
        The model will predict the customer segment and show a business recommendation.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        total_spent = st.number_input("Total Spent", min_value=0.0, value=50000.0)
        avg_check = st.number_input("Average Check", min_value=0.0, value=650.0)

    with col2:
        num_checks = st.number_input("Number of Transactions", min_value=0, value=65)
        total_items = st.number_input("Total Items", min_value=0, value=900)

    with col3:
        avg_items_per_check = st.number_input("Average Items per Transaction", min_value=0.0, value=13.0)
        recency = st.number_input("Recency, days since last purchase", min_value=0, value=3)

    if st.button("🔍 Predict Customer Segment"):
        input_data = pd.DataFrame([{
            "total_spent": total_spent,
            "avg_check": avg_check,
            "num_checks": num_checks,
            "total_items": total_items,
            "avg_items_per_check": avg_items_per_check,
            "recency": recency
        }])

        scaled_data = sc.transform(input_data)
        cluster = km.predict(scaled_data)[0]

        segment = (
            cluster_names
            .set_index("cluster")
            .loc[cluster, "segment_name"]
        )

        info = customer_descriptions.get(segment, {
            "emoji": "🎯",
            "description": "Segment description is not available.",
            "recommendation": "Analyze this group manually."
        })

        st.markdown("---")

        left, right = st.columns([1, 2])

        with left:
            st.metric("Predicted Cluster", int(cluster))
            st.metric("Customer Segment", segment)

        with right:
            st.success(f"{info['emoji']} **{segment}**")
            st.write(f"**Description:** {info['description']}")
            st.info(f"**Manager recommendation:** {info['recommendation']}")

        st.subheader("📊 Segment Profile")
        st.dataframe(cluster_summary[cluster_summary["cluster"] == cluster])

        st.subheader("🧾 Input Data")
        st.dataframe(input_data)


with tab2:
    st.header("📦 Product Segmentation")

    if not has_products:
        st.warning(
            "Product segmentation files were not found. "
            "Add product_segments.csv and product_cluster_summary.csv to enable this tab."
        )

    else:
        st.markdown(
            """
            This section shows product segments based on revenue, quantity sold,
            number of transactions and customer reach.
            """
        )

        st.subheader("📊 Product Segment Summary")
        st.dataframe(product_cluster_summary)

        segment_col = None
        for possible_name in ["product_segment", "segment_name", "segment"]:
            if possible_name in product_segments.columns:
                segment_col = possible_name
                break

        if segment_col:
            selected_segment = st.selectbox(
                "Choose product segment",
                product_segments[segment_col].dropna().unique()
            )

            info = product_descriptions.get(selected_segment, {
                "emoji": "📦",
                "description": "Product segment description is not available.",
                "recommendation": "Analyze this segment manually."
            })

            st.success(f"{info['emoji']} **{selected_segment}**")
            st.write(f"**Description:** {info['description']}")
            st.info(f"**Business recommendation:** {info['recommendation']}")

            st.subheader("Top Products in This Segment")

            filtered = product_segments[product_segments[segment_col] == selected_segment]

            if "total_revenue" in filtered.columns:
                filtered = filtered.sort_values("total_revenue", ascending=False)

            st.dataframe(filtered.head(15))

        else:
            st.dataframe(product_segments.head(20))
with tab3:
    st.header("🎯 Personalized Marketing Recommendation")

    customer_id = st.number_input(
        "Enter Customer ID",
        min_value=1,
        step=1
    )

    if st.button("Get Recommendation"):
        row = recommendations[recommendations["customer_id"] == customer_id]

        if row.empty:
            st.warning("Customer not found")
        else:
            row = row.iloc[0]

            st.success(f"Segment: {row['segment_name']}")
            st.info(f"Favorite category: {row['favorite_category']}")
            st.write(f"💡 Recommended campaign: **{row['marketing_recommendation']}**")
            st.subheader("🛒 Recommended products")

            if "recommended_products" in row.index and pd.notna(row["recommended_products"]):
                products = ast.literal_eval(row["recommended_products"])

                if len(products) == 0:
                    st.warning("No product recommendations found")
                else:
                    for product in products:
                        st.write(f"• {product}")
            else:
                st.warning("Recommended products column not found")

            customer_categories = (
                category_summary[category_summary["CustomerID"] == customer_id]
                .sort_values("category_spent", ascending=False)
            )

            st.subheader("🧾 Customer purchases by category")

            if customer_categories.empty:
                st.warning("No category data found for this customer")
            else:
                st.dataframe(
                    customer_categories.rename(columns={
                        "CategoryName": "Category",
                        "category_spent": "Total spent in category",
                        "category_items": "Items bought",
                        "category_transactions": "Transactions"
                    }),
                    use_container_width=True
                )

