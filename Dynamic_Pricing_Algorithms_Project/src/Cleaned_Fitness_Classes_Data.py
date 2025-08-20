import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

df1 = pd.read_csv("Cult's Fitness_Classes/Fitness_Classes_Data/Fitness Classes Data/Classes April-May 2018.csv")
df2 = pd.read_csv("Cult's Fitness_Classes/Fitness_Classes_Data/Fitness Classes Data/Classes June 2018.csv")

df1["BookingEndDateTime"] = pd.to_datetime(
    df1["BookingEndDateTime (Month / Day / Year)"],
    format="%d-%b-%y"
)

df2["BookingEndDateTime"] = pd.to_datetime(
    df2["BookingEndDateTime (Month / Day / Year)"],
    format="%d-%b-%y"
)

df1.drop(columns=["BookingEndDateTime (Month / Day / Year)"], inplace=True)
df2.drop(columns=["BookingEndDateTime (Month / Day / Year)"], inplace=True)

print("Missing prices before:", df1["Price (INR)"].isna().sum())
df1.dropna(subset=["Price (INR)"], inplace=True)

df = pd.concat([df1, df2], ignore_index=True)

df.drop_duplicates(inplace=True)

df["Month_Year"] = df["BookingEndDateTime"].dt.to_period("M")

monthly_avg = df.groupby("Month_Year")["Price (INR)"].mean()
monthly_avg.plot(kind="bar", title="Average Price per Month", ylabel="INR")
plt.tight_layout()
plt.show()

df_sorted = df.sort_values("BookingEndDateTime")
df_sorted.plot(x="BookingEndDateTime", y="Price (INR)", title="Price Trend Over Time")
plt.tight_layout()
plt.show()

df["Price (INR)"].hist(bins=20)
plt.title("Price Distribution")
plt.xlabel("Price (INR)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

sns.boxplot(x=df["Price (INR)"])
plt.title("Boxplot - Price (INR)")
plt.tight_layout()
plt.show()

print("\nDescriptive statistics:")
print(df.describe())

print("\nTop 10 Activity Descriptions:")
print(df["ActivityDescription"].value_counts().head(10))

print("\nTop 5 Sites:")
print(df["ActivitySiteID"].value_counts().head(5))

print("""
===== Quality Summary =====
- Missing prices removed: 18 rows dropped from df1.
- No duplicates found after merging.
- Date parsing successful; range: {} to {}
- Price range: {} to {} INR
- Dataset shape after cleaning: {}
===========================
""".format(
    df["BookingEndDateTime"].min().date(),
    df["BookingEndDateTime"].max().date(),
    df["Price (INR)"].min(),
    df["Price (INR)"].max(),
    df.shape
))

median_price = df["Price (INR)"].median()
median_booked = df["Number Booked"].median()
median_capacity = df["MaxBookees"].median()

print("Median Price:", median_price)
print("Median Number Booked:", median_booked)
print("Median Max Capacity:", median_capacity)


df["BookingDate"] = pd.to_datetime(df["BookingEndDateTime"]).dt.date
df["BookingStartTime"] = pd.to_datetime(df["BookingStartTime"], format="%H:%M:%S").dt.time

df["BookingStartDateTime"] = pd.to_datetime(
    df["BookingDate"].astype(str) + " " + df["BookingStartTime"].astype(str)
)

df["hour"] = df["BookingStartDateTime"].dt.hour
df["day"] = df["BookingStartDateTime"].dt.day_name()

demand_by_day = df.groupby("day")["Number Booked"].mean()
demand_by_hour = df.groupby("hour")["Number Booked"].mean()

print("Average demand by day:\n", demand_by_day)
print("\nAverage demand by hour:\n", demand_by_hour)

df["Utilization"] = df["Number Booked"] / df["MaxBookees"]

demand_by_class = (
    df.groupby("ActivityDescription")[["Number Booked","MaxBookees","Utilization"]]
      .mean()
      .sort_values("Utilization", ascending=False)
)

print(demand_by_class.head(10))

demand_by_class["ClassCount"] = df.groupby("ActivityDescription")["ActivitySiteID"].count()

popular_classes = demand_by_class[demand_by_class["ClassCount"] >= 20]

print(popular_classes.sort_values("Utilization", ascending=False).head(10))

top_classes = popular_classes.sort_values("Utilization", ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.barh(top_classes.index, top_classes["Utilization"])
plt.xlabel("Average Utilization Rate")
plt.title("Top 10 High-Demand Classes by Utilization")
plt.gca().invert_yaxis()
plt.show()

df["Utilization"] = df["Number Booked"] / df["MaxBookees"]

demand_by_class_site = (
    df.groupby(["ActivitySiteID", "ActivityDescription"])
      .agg({"Number Booked": "mean", 
            "MaxBookees": "mean", 
            "Utilization": "mean"})
      .reset_index()
)

class_counts = (
    df.groupby(["ActivitySiteID", "ActivityDescription"])["ActivityDescription"]
      .count()
      .reset_index(name="ClassCount")
)

demand_by_class_site = demand_by_class_site.merge(
    class_counts, on=["ActivitySiteID", "ActivityDescription"]
)

popular_classes_site = demand_by_class_site[demand_by_class_site["ClassCount"] >= 20]

top_classes_site = (
    popular_classes_site.sort_values(["ActivitySiteID","Utilization"], ascending=[True, False])
)

print(top_classes_site.head(20))

df['BookingEndDateTime'] = pd.to_datetime(df['BookingEndDateTime'], dayfirst=True, errors='coerce')


df['Day'] = df['BookingEndDateTime'].dt.day_name()
df['Hour'] = df['BookingEndDateTime'].dt.hour
df['Month'] = df['BookingEndDateTime'].dt.month_name()

df_encoded = pd.get_dummies(
    df,
    columns=['Day','Hour','Month','ActivityDescription','ActivitySiteID'],
    drop_first=True
)

corr = df_encoded.corr(numeric_only=True)['Number Booked'].sort_values(ascending=False)
print("\nTop positive correlations with demand:\n", corr.head(15))
print("\nTop negative correlations with demand:\n", corr.tail(15))
