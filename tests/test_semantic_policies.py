from app.context.semantic_policies import build_semantic_policy_section


def test_build_semantic_policy_section_formats_one_block_per_table() -> None:
    section = build_semantic_policy_section(
        ["olist_orders_dataset", "carriers"],
        [["ESTADO: regla uno."], ["PUNTUALIDAD: regla dos."]],
    )

    assert "olist_orders_dataset:" in section
    assert "  - ESTADO: regla uno." in section
    assert "carriers:" in section
    assert "  - PUNTUALIDAD: regla dos." in section


def test_build_semantic_policy_section_skips_tables_without_policies() -> None:
    section = build_semantic_policy_section(
        ["olist_sellers_dataset", "olist_orders_dataset"],
        [[], ["ESTADO: regla."]],
    )

    assert "olist_sellers_dataset" not in section
    assert "olist_orders_dataset:" in section


def test_build_semantic_policy_section_returns_empty_string_when_nothing_qualifies() -> None:
    assert build_semantic_policy_section([], []) == ""
    assert build_semantic_policy_section(["olist_sellers_dataset"], [[]]) == ""


def test_build_semantic_policy_section_lists_multiple_policies_per_table() -> None:
    section = build_semantic_policy_section(
        ["olist_orders_dataset"],
        [["TEMPORAL: uno.", "ESTADO: dos."]],
    )

    assert section.count("  - ") == 2
