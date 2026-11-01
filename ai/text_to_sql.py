
import os
import re
import json
import pandas as pd
import streamlit as st
import snowflake.connector

from google import genai
from dotenv import load_dotenv

load_dotenv()

# Gemini configuration
MODEL = "gemini-3.8-flash"

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Example questions
EXAMPLE_QUESTIONS = [
    "Top 10 cities by GMV",
    "Which cuisine has the most orders?",
    "Average delivery time by city, worst first",
    "Cancel rate by payment method"
]

# Snowflake schema information
SCHEMA = """
Tables available in Snowflake. Use bare table names,
without database or schema prefixes.

FCT_ORDERS(
    order_id, order_date, customer_id, restaurant_id, city, cuisine,
    payment_method, order_status, is_delivered, sales_amount,
    discount, delivery_fee, gst, customer_rating, delivery_time_min
)

DIM_RESTAURANT(
    restaurant_id, restaurant_name, city, cuisine, rating, cost_for_two
)

DIM_CUSTOMER(
    customer_id, customer_name, age, age_segment, gender, city
)

MART_DAILY_CITY_REVENUE(
    order_date, city, orders, cancel_rate, gmv, aov
)

MART_RESTAURANT_PERFORMANCE(
    restaurant_id, restaurant_name, city, cuisine,
    orders, revenue, avg_customer_rating, cancel_rate
)

MART_DELIVERY_SLA(
    city, order_hour, delivered_orders, p50_delivery_min, late_rate
)

Notes:
- GMV means delivered revenue.
- Prefer MART tables when they fit the question.
- Use only the tables and columns listed above.
"""

SYSTEM_PROMPT = f"""
You are an expert Snowflake SQL developer.

Convert the user's natural-language question into ONE read-only SQL query.

Rules:
1. Generate SELECT queries only.
2. WITH clauses are allowed only when the final statement is SELECT.
3. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
   TRUNCATE, MERGE, GRANT, REVOKE or other modifying statements.
4. Use only the tables and columns provided in the schema.
5. Use bare table names without database or schema prefixes.
6. Add LIMIT 100 or less unless the query returns a single aggregate result.
7. Do not invent columns or tables.
8. Return valid JSON only, in this format:
   {{"sql": "SELECT ..."}}

Database schema:
{SCHEMA}
"""


# 1. Snowflake connection
@st.cache_resource
def get_connection():

    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema="MARTS",
        role="DBT_ROLE"
    )


# 2. Generate SQL using Gemini
def generate_sql(question):

    response = client.models.generate_content(
        model=MODEL,
        contents=f"""
        {SYSTEM_PROMPT}

        User question:
        {question}
        """,
        config={
            "temperature": 0,
            "response_mime_type": "application/json"
        }
    )

    answer = json.loads(response.text)

    sql = answer["sql"].strip()

    # Remove SQL markdown formatting if present
    sql = re.sub(r"^```sql\s*|^```\s*|```$", "", sql, flags=re.IGNORECASE)
    sql = sql.strip().rstrip(";")

    return sql


# 3. Validate SQL safety
def is_safe(sql):

    # Remove comments before checking
    cleaned = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    cleaned = re.sub(r"--.*?$", "", cleaned, flags=re.MULTILINE)

    # Only SELECT or WITH queries
    if not re.match(r"^\s*(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return False

    # Reject multiple SQL statements
    if ";" in cleaned:
        return False

    # Reject modifying SQL keywords
    forbidden = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|"
        r"REPLACE|MERGE|GRANT|REVOKE|CALL|EXECUTE|COPY|PUT|"
        r"REMOVE|UNDROP|USE)\b",
        re.IGNORECASE
    )

    if forbidden.search(cleaned):
        return False

    # Reject SELECT INTO
    if re.search(r"\bINTO\b", cleaned, re.IGNORECASE):
        return False

    return True


# 4. Execute SQL in Snowflake
def run_query(sql):

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
        return cursor.fetch_pandas_all()

    finally:
        cursor.close()


# 5. Streamlit interface
st.title("Chat with your Zomato Data")

st.caption(
    f"Ask questions in English. Gemini ({MODEL}) "
    "generates SQL and Snowflake executes it."
)

with st.sidebar:

    st.header("Example Questions")

    for q in EXAMPLE_QUESTIONS:
        st.markdown(f"- {q}")


question = st.text_input(
    "Enter your question here",
    placeholder="e.g. Top 10 restaurants by revenue in Bangalore"
)


if question:

    try:

        with st.spinner("Gemini is generating SQL..."):

            sql = generate_sql(question)

        st.subheader("Generated SQL")
        st.code(sql, language="sql")

        if not is_safe(sql):

            st.error(
                "The generated SQL is not safe to execute. "
                "Please try another question."
            )

        else:

            with st.spinner("Executing query in Snowflake..."):

                df = run_query(sql)

            st.success(f"{len(df)} rows returned")

            st.dataframe(df, hide_index=True)

            # Display bar chart for two-column numeric results
            if (
                len(df.columns) == 2
                and pd.api.types.is_numeric_dtype(df.iloc[:, 1])
            ):

                st.subheader("Visualisation")

                st.bar_chart(
                    df,
                    x=df.columns[0],
                    y=df.columns[1]
                )

    except Exception as e:

        st.error(f"Something went wrong: {e}")
