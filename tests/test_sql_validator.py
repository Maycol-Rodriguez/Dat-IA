from langchain_community.utilities import SQLDatabase

from app.validation.sql_validator import validate_sql


def _sqlite_db_with_carriers() -> SQLDatabase:
    db = SQLDatabase.from_uri("sqlite:///:memory:")
    db.run("CREATE TABLE carriers (carrier_name TEXT, on_time_rate REAL);")
    return db


def test_validate_sql_accepts_query_with_allowed_tables() -> None:
    sql = (
        "SELECT carrier_name FROM carriers "
        "ORDER BY on_time_rate DESC LIMIT 1;"
    )

    result = validate_sql(sql, allowed_tables=["carriers"])

    assert result.is_valid is True
    assert result.stage == "ok"
    assert result.error == ""
    assert "LIMIT 1" in result.sql


def test_validate_sql_accepts_join_with_multiple_allowed_tables() -> None:
    sql = (
        "SELECT o.order_id, oi.price "
        "FROM olist_orders_dataset o "
        "JOIN olist_order_items_dataset oi ON o.order_id = oi.order_id;"
    )

    result = validate_sql(
        sql,
        allowed_tables=[
            "olist_orders_dataset",
            "olist_order_items_dataset",
        ],
    )

    assert result.is_valid is True
    assert result.stage == "ok"


def test_validate_sql_accepts_query_without_tables() -> None:
    result = validate_sql(
        "SELECT 1 AS prototype_result;",
        allowed_tables=[],
    )

    assert result.is_valid is True
    assert result.stage == "ok"


def test_validate_sql_rejects_invalid_syntax() -> None:
    result = validate_sql("SELECT FROM WHERE;", allowed_tables=["carriers"])

    assert result.is_valid is False
    assert result.stage == "syntax"
    assert result.error


def test_validate_sql_rejects_empty_sql() -> None:
    result = validate_sql("   ", allowed_tables=["carriers"])

    assert result.is_valid is False
    assert result.stage == "syntax"


def test_validate_sql_rejects_table_outside_retrieved_schema() -> None:
    sql = "SELECT * FROM customer_support_tickets;"

    result = validate_sql(sql, allowed_tables=["carriers"])

    assert result.is_valid is False
    assert result.stage == "tables"
    assert "customer_support_tickets" in result.error


def test_validate_sql_does_not_flag_cte_alias_as_unknown_table() -> None:
    sql = """
    WITH ranked AS (
        SELECT carrier_name, on_time_rate FROM carriers
    )
    SELECT * FROM ranked;
    """

    result = validate_sql(sql, allowed_tables=["carriers"])

    assert result.is_valid is True
    assert result.stage == "ok"


def test_validate_sql_dry_run_passes_with_valid_columns() -> None:
    db = _sqlite_db_with_carriers()
    sql = "SELECT carrier_name FROM carriers ORDER BY on_time_rate DESC;"

    result = validate_sql(sql, allowed_tables=["carriers"], db=db)

    assert result.is_valid is True
    assert result.stage == "ok"


def test_validate_sql_dry_run_rejects_unknown_column() -> None:
    db = _sqlite_db_with_carriers()
    sql = "SELECT on_time_percentage FROM carriers;"

    result = validate_sql(sql, allowed_tables=["carriers"], db=db)

    assert result.is_valid is False
    assert result.stage == "dry_run"
    assert result.error


def test_validate_sql_skips_dry_run_when_db_is_none() -> None:
    sql = "SELECT on_time_percentage FROM carriers;"

    result = validate_sql(sql, allowed_tables=["carriers"], db=None)

    assert result.is_valid is True
    assert result.stage == "ok"


def test_validate_sql_rejects_stacked_statements() -> None:
    sql = "SELECT * FROM carriers; DROP TABLE carriers;"

    result = validate_sql(sql, allowed_tables=["carriers"])

    assert result.is_valid is False
    assert result.stage == "statement"
    assert result.error == "Solo se permite una sentencia SQL por consulta."


def test_validate_sql_rejects_non_select_statement() -> None:
    sql = "DROP TABLE carriers;"

    result = validate_sql(sql, allowed_tables=["carriers"])

    assert result.is_valid is False
    assert result.stage == "statement"
    assert result.error == "Solo se permiten sentencias SELECT."


def test_validate_sql_injects_limit_when_missing() -> None:
    sql = "SELECT carrier_name FROM carriers;"

    result = validate_sql(sql, allowed_tables=["carriers"])

    assert result.is_valid is True
    assert "LIMIT 200" in result.sql


def test_validate_sql_caps_limit_above_max_rows() -> None:
    sql = "SELECT carrier_name FROM carriers LIMIT 5000;"

    result = validate_sql(sql, allowed_tables=["carriers"], max_rows=200)

    assert result.is_valid is True
    assert "LIMIT 200" in result.sql
    assert "LIMIT 5000" not in result.sql


def test_validate_sql_preserves_limit_below_max_rows() -> None:
    sql = "SELECT carrier_name FROM carriers LIMIT 5;"

    result = validate_sql(sql, allowed_tables=["carriers"], max_rows=200)

    assert result.is_valid is True
    assert "LIMIT 5" in result.sql
    assert "LIMIT 200" not in result.sql


def test_validate_sql_treats_limit_all_as_missing() -> None:
    sql = "SELECT carrier_name FROM carriers LIMIT ALL;"

    result = validate_sql(sql, allowed_tables=["carriers"], max_rows=200)

    assert result.is_valid is True
    assert "LIMIT 200" in result.sql


def test_validate_sql_dry_run_uses_bounded_sql() -> None:
    db = _sqlite_db_with_carriers()
    sql = "SELECT carrier_name FROM carriers ORDER BY on_time_rate DESC;"

    result = validate_sql(sql, allowed_tables=["carriers"], db=db, max_rows=1)

    assert result.is_valid is True
    assert "LIMIT 1" in result.sql
