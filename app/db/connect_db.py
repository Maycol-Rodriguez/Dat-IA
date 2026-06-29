import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

def execute_query(query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, engine)

# Ejemplo de uso

# query = """
# SELECT *
# FROM olist_customers_dataset
# LIMIT 5;
# """

# df = execute_query(query)

# df.head()