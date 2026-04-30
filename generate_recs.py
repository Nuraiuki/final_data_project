import pandas as pd
import numpy as np

print("Reading datasets...")
df = pd.read_csv("df_merged.csv", usecols=["CustomerID", "ProductID", "ProductName", "CategoryName", "Revenue", "Quantity"])
customers_marketing = pd.read_csv("customer_marketing_recommendations.csv")

print("Calculating top products...")
top_products = (
    df.groupby(["CategoryName", "ProductID", "ProductName"])
      .agg(
          total_revenue=("Revenue", "sum"),
          total_quantity=("Quantity", "sum"),
          num_customers=("CustomerID", "nunique")
      )
      .reset_index()
)

top_products_ranked = (
    top_products
    .sort_values(["CategoryName", "total_revenue"], ascending=[True, False])
    .groupby("CategoryName")
    .head(5)
)

def recommend_products(favorite_category):
    if pd.isna(favorite_category):
        return []
    
    recs = top_products_ranked[
        top_products_ranked["CategoryName"] == favorite_category
    ]["ProductName"].tolist()
    
    return recs

print("Applying recommendations...")
customers_marketing["recommended_products"] = customers_marketing["favorite_category"].apply(recommend_products)

print("Saving to customer_recommendations.csv...")
customers_marketing.to_csv("customer_recommendations.csv", index=False)
print("Done!")
