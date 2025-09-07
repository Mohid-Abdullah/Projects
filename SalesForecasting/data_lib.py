from pyspark.sql import SparkSession  
from pyspark.sql.functions import col, lit, lower, when, to_date, sum,expr,coalesce, trim, regexp_replace
import warnings

class DataInitializer:
    def __init__(self,fileName:str,sessionName:str,options:dict):

        if not fileName:
            raise ValueError("error: fileName is empty")
        if not sessionName:
            raise ValueError("error: sessionName is empty")
        if not isinstance(options, dict):
            raise TypeError("error: options must be a dictionary")
        
        session = SparkSession.builder.appName(sessionName).master("local[*]").config("spark.sql.ansi.enabled", "false").getOrCreate()

        c_fileName = fileName
        df = session.read.options(**options).csv(c_fileName)
        self.df = session.read.options(**options).csv(fileName)

def readSparkDf(fileName: str, session: SparkSession = "none"):
    session = SparkSession.builder.appName(session).master("local[*]").config("spark.sql.ansi.enabled", "false").getOrCreate()
    df = session.read.options(inferSchema="true", header="true").csv(fileName)
    dfpd = df.toPandas()
    return dfpd

def nullCount(df):
    warnings.warn(
            "WARNING: nullCount collects results to the driver. "
            "Avoid if more than 100 coloumns",
            UserWarning
    )
     
    nullCols = df.select([sum(when(col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns])
    nullDict = nullCols.collect()[0].asDict()
    for a,b in nullDict.items():
        if b == 0:
            continue
        else:
            print(f"{a}:{b}")

def parse_date(colomn_name: str):
    c = trim(col(colomn_name))
    c = regexp_replace(c, r'[\uFEFF\u200B]', '')
    c = regexp_replace(c, r'^"|"$', '')
    c = regexp_replace(c, r'-', '/')
    return coalesce(
        to_date(c, "M/d/yyyy"),
        to_date(c, "MM/dd/yyyy"),
        to_date(c, "d/M/yyyy"),
        to_date(c, "dd/MM/yyyy"),
        to_date(c, "yyyy/M/d"),
        to_date(c, "yyyy-MM-dd")
    )
