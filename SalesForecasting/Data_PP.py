import warnings
warnings.filterwarnings("ignore")
from colorama import init
from pyspark.sql import DataFrameNaFunctions, SparkSession  
from pyspark.sql.functions import col, lit, lower, when, to_date, sum,expr,coalesce
from pyspark.sql.window import Window
from pyspark.ml.feature import Imputer
from data_lib import DataInitializer, nullCount, parse_date

df_init = DataInitializer(
    "Sample - Superstore.csv",
    "Superstore",
    {
        "inferSchema": "true",
        "header": "true"
    }
)

df = df_init.df
print("Original row count:", df.count())
print(nullCount(df)) #as nullcount is zero ill just do the basic cleaning

df = df.withColumn("Sales", expr("try_cast(Sales as double)")) \
       .withColumn("Quantity", expr("try_cast(Quantity as int)")) \
       .withColumn("Discount", expr("try_cast(Discount as double)"))

df = df.filter(col("Order Date").isNotNull() & 
               col("Ship Date").isNotNull() & 
               (col("Order Date") != "") & 
               (col("Ship Date") != ""))

df = df.withColumn("Order Date", parse_date("Order Date")) \
       .withColumn("Ship Date",  parse_date("Ship Date"))

df.printSchema()
df = df.filter((col("Discount") <= 0.80))
df = df.dropna()
print("Final row count:", df.count())

df1 = df.select(['Quantity', 'Category', 'Discount', 'Region','Segment' , 'Ship Mode', 'State','Sub-Category','Sales'])
df1.write.csv("Cleaned/Sales_cleaned.csv", header=True, mode="overwrite")
