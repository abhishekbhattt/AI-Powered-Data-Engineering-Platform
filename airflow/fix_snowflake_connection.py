
import os
from airflow.models import Connection
from airflow.utils.session import create_session

connection = Connection(
    conn_id="snowflake_default",
    conn_type="snowflake",
    login=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    schema="RAW",
    extra={
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "warehouse": "ZOMATO_WH",
        "database": "ZOMATO",
        "role": "DBT_ROLE"
    }
)

with create_session() as session:
    old_connection = (
        session.query(Connection)
        .filter(Connection.conn_id == "snowflake_default")
        .first()
    )

    if old_connection:
        session.delete(old_connection)

    session.add(connection)

print("Snowflake connection created successfully!")
