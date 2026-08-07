"""Formatea las políticas semánticas de las tablas recuperadas para el prompt
del generador SQL.

Las políticas viven en el campo `politicas` de cada entrada de
`data/ddl.json`, separadas de `descripcion` (que solo se vectoriza para
retrieval). Este módulo no decide qué políticas aplican — eso ya lo resolvió
la recuperación al elegir qué tablas traer — solo las da vuelta a texto.
"""

from __future__ import annotations


def build_semantic_policy_section(
    tables: list[str],
    policies_by_table: list[list[str]],
) -> str:
    """Arma la sección de políticas para las tablas ya recuperadas.

    `tables` y `policies_by_table` van alineados por índice, igual que
    `EmbeddingsResponse.tabla` y `EmbeddingsResponse.descripcion`. Las
    tablas sin políticas no generan ninguna línea.
    """
    sections = []

    for table, table_policies in zip(tables, policies_by_table):
        cleaned_policies = [policy.strip() for policy in table_policies if policy.strip()]

        if not cleaned_policies:
            continue

        bullet_list = "\n".join(f"  - {policy}" for policy in cleaned_policies)
        sections.append(f"{table}:\n{bullet_list}")

    return "\n".join(sections)
