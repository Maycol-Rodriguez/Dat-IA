"""Generador de tablas sinteticas para el esquema Olist Extended

Lee los CSV originales de Olist en ``data/`` y produce, en ``data/data_sintetica/``:
  - 7 tablas sinteticas: carriers, product_price_history, seller_promotions,
    warehouse_inventory, delivery_incidents, customer_support_tickets, product_returns.
  - 2 tablas originales extendidas con sus FK nuevas: olist_order_items_extended,
    olist_order_payments_extended.

La generacion es reproducible (SEED fijo) y referencialmente coherente con la data real:
toda FK apunta solo a PKs existentes y las fechas caen dentro del rango observado en Olist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- Configuracion ---------------------------------------------------------
SEED: int = 42
BASE_DIR: Path = Path(__file__).resolve().parents[1] / "data"
OUT_DIR: Path = BASE_DIR / "data_sintetica"

# Volumenes objetivo (parametrizados para ajuste facil)
N_CARRIERS: int = 25
N_WAREHOUSES: int = 12
N_PRICE_PAIRS: int = 3_500          # pares producto-vendedor con historial de precio
N_PROMOTIONS: int = 5_000
N_INVENTORY_PAIRS: int = 18_000     # pares con registro de inventario
N_INCIDENTS: int = 10_000
N_TICKETS: int = 13_000
N_RETURNS: int = 7_000

# Estados de Brasil (27)
ESTADOS_BR: list[str] = [
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO",
]

rng = np.random.default_rng(SEED)


# --- Utilidades ------------------------------------------------------------
def gen_uuids(n: int) -> np.ndarray:
    """Genera ``n`` strings con formato UUID deterministas a partir del RNG sembrado."""
    hexchars = np.array(list("0123456789abcdef"))
    digits = hexchars[rng.integers(0, 16, size=(n, 32))]
    s = ["".join(row) for row in digits]
    return np.array([f"{x[:8]}-{x[8:12]}-{x[12:16]}-{x[16:20]}-{x[20:]}" for x in s])


def weighted_sample_idx(weights: np.ndarray, k: int) -> np.ndarray:
    """Muestra ``k`` indices sin reemplazo proporcional a ``weights``."""
    p = weights / weights.sum()
    k = min(k, len(weights))
    return rng.choice(len(weights), size=k, replace=False, p=p)


# --- Carga de originales ---------------------------------------------------
def cargar_originales() -> dict[str, pd.DataFrame]:
    """Lee los CSV originales necesarios y normaliza dtypes de llaves a str."""
    orders = pd.read_csv(
        BASE_DIR / "olist_orders_dataset.csv",
        usecols=[
            "order_id", "customer_id", "order_status",
            "order_purchase_timestamp", "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    customers = pd.read_csv(
        BASE_DIR / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_state"],
    )
    items = pd.read_csv(BASE_DIR / "olist_order_items_dataset.csv")
    payments = pd.read_csv(BASE_DIR / "olist_order_payments_dataset.csv")
    reviews = pd.read_csv(
        BASE_DIR / "olist_order_reviews_dataset.csv",
        usecols=["order_id", "review_score"],
    )
    sellers = pd.read_csv(
        BASE_DIR / "olist_sellers_dataset.csv",
        usecols=["seller_id", "seller_state"],
    )

    # Fechas
    for col in ("order_purchase_timestamp", "order_delivered_customer_date",
                "order_estimated_delivery_date"):
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"], errors="coerce")

    # Llaves como str (CLAUDE: alinear dtype antes de cualquier merge)
    for df, cols in (
        (orders, ["order_id", "customer_id"]),
        (customers, ["customer_id"]),
        (items, ["order_id", "product_id", "seller_id"]),
        (payments, ["order_id"]),
        (reviews, ["order_id"]),
        (sellers, ["seller_id"]),
    ):
        for c in cols:
            df[c] = df[c].astype(str)

    return {
        "orders": orders, "customers": customers, "items": items,
        "payments": payments, "reviews": reviews, "sellers": sellers,
    }


# --- 1. carriers -----------------------------------------------------------
def gen_carriers() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Genera transportistas y un mapa estado -> carriers que lo cubren (>=3 por estado)."""
    nombres = [
        "Transportadora Andina", "LogExpress BR", "RapidoSul", "EntregaJa",
        "Correios Plus", "MercadoEnvios", "FreteFacil", "VelozCargo",
        "NordesteLog", "AtlanticoTransportes", "PampaExpress", "CentroOeste Cargas",
        "TotalEntregas", " AgilBR", "PremiumLog", "EcoFrete", "MetropoleEnvios",
        "InterEstadual Cargo", "BrasilLog", "TurboEntrega", "SafeShip BR",
        "AmazoniaCargas", "CostaVerde Logistica", "SerraExpress", "UniLog Brasil",
    ][:N_CARRIERS]

    tipos = rng.choice(
        ["express", "standard", "economy", "freight"],
        size=N_CARRIERS, p=[0.25, 0.4, 0.25, 0.1],
    )
    params = {  # tipo -> (dias_min, dias_max, ontime_min, ontime_max, costo_min, costo_max)
        "express":  (2, 5, 0.90, 0.99, 8.0, 18.0),
        "standard": (5, 10, 0.78, 0.92, 4.0, 9.0),
        "economy":  (9, 18, 0.62, 0.82, 2.0, 5.0),
        "freight":  (7, 20, 0.70, 0.88, 1.5, 4.0),
    }
    avg_days, ontime, costo = [], [], []
    for t in tipos:
        dmin, dmax, omin, omax, cmin, cmax = params[t]
        avg_days.append(round(float(rng.uniform(dmin, dmax)), 1))
        ontime.append(round(float(rng.uniform(omin, omax)), 3))
        costo.append(round(float(rng.uniform(cmin, cmax)), 2))

    carrier_ids = gen_uuids(N_CARRIERS)

    # Cobertura: cada carrier cubre entre 5 y 20 estados
    coberturas: list[list[str]] = []
    for _ in range(N_CARRIERS):
        n = int(rng.integers(5, 21))
        coberturas.append(sorted(rng.choice(ESTADOS_BR, size=n, replace=False).tolist()))

    # Garantizar >=3 carriers por estado
    cobertura_set = [set(c) for c in coberturas]
    for est in ESTADOS_BR:
        cubren = [i for i in range(N_CARRIERS) if est in cobertura_set[i]]
        faltan = 3 - len(cubren)
        if faltan > 0:
            candidatos = [i for i in range(N_CARRIERS) if est not in cobertura_set[i]]
            elegidos = rng.choice(candidatos, size=min(faltan, len(candidatos)), replace=False)
            for i in np.atleast_1d(elegidos):
                cobertura_set[int(i)].add(est)

    coverage_regions = [",".join(sorted(s)) for s in cobertura_set]
    mapa_estado: dict[str, list[str]] = {
        est: [carrier_ids[i] for i in range(N_CARRIERS) if est in cobertura_set[i]]
        for est in ESTADOS_BR
    }

    carriers = pd.DataFrame({
        "carrier_id": carrier_ids,
        "carrier_name": nombres,
        "carrier_type": tipos,
        "avg_delivery_days": avg_days,
        "coverage_regions": coverage_regions,
        "cost_per_kg": costo,
        "on_time_rate": ontime,
    })
    return carriers, mapa_estado


# --- 2. product_price_history ---------------------------------------------
def gen_price_history(items: pd.DataFrame) -> pd.DataFrame:
    """Historial de cambios de precio sobre un subconjunto de pares producto-vendedor."""
    precio_pair = items.groupby(["product_id", "seller_id"], as_index=False).agg(
        precio_ref=("price", "median"))
    elegidos = precio_pair.iloc[weighted_sample_idx(np.ones(len(precio_pair)), N_PRICE_PAIRS)]

    razones = ["ajuste_inflacion", "promocion", "cambio_proveedor",
               "correccion_precio", "demanda_estacional"]
    t0 = pd.Timestamp("2016-09-01")
    span_dias = (pd.Timestamp("2018-10-01") - t0).days

    pids, sids, olds, news, fechas, motivos, eids = [], [], [], [], [], [], []
    prod = elegidos["product_id"].to_numpy()
    sell = elegidos["seller_id"].to_numpy()
    ref = elegidos["precio_ref"].to_numpy()
    for i in range(len(elegidos)):
        n_ev = int(rng.integers(1, 5))
        # precio inicial cercano a la referencia observada
        precio_actual = max(0.5, float(ref[i]) * float(rng.uniform(0.8, 1.2)))
        dias = np.sort(rng.integers(0, span_dias, size=n_ev))
        for d in dias:
            delta = float(rng.uniform(-0.2, 0.25))
            nuevo = max(0.5, round(precio_actual * (1 + delta), 2))
            pids.append(prod[i]); sids.append(sell[i])
            olds.append(round(precio_actual, 2)); news.append(nuevo)
            fechas.append(t0 + pd.Timedelta(days=int(d)) + pd.Timedelta(hours=int(rng.integers(0, 24))))
            motivos.append(rng.choice(razones))
            precio_actual = nuevo
    eids = gen_uuids(len(pids))

    return pd.DataFrame({
        "price_event_id": eids,
        "product_id": pids,
        "seller_id": sids,
        "old_price": olds,
        "new_price": news,
        "change_at": fechas,
        "change_reason": motivos,
    })


# --- 3. seller_promotions --------------------------------------------------
def gen_promotions(items: pd.DataFrame) -> pd.DataFrame:
    """Promociones de vendedores sobre productos de su catalogo."""
    pares = items[["seller_id", "product_id"]].drop_duplicates().reset_index(drop=True)
    idx = rng.integers(0, len(pares), size=N_PROMOTIONS)  # con reemplazo (varias promos por par)
    sel = pares.iloc[idx].reset_index(drop=True)

    discount = np.round(rng.uniform(0.05, 0.50, size=N_PROMOTIONS), 2)
    t0 = pd.Timestamp("2016-09-01")
    span_dias = (pd.Timestamp("2018-09-15") - t0).days
    inicio = t0 + pd.to_timedelta(rng.integers(0, span_dias, size=N_PROMOTIONS), unit="D")
    duracion = rng.integers(7, 31, size=N_PROMOTIONS)
    fin = inicio + pd.to_timedelta(duracion, unit="D")
    promo_type = rng.choice(
        ["flash_sale", "seasonal", "clearance", "bundle", "loyalty"],
        size=N_PROMOTIONS, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    # unidades vendidas crecen con el descuento
    units = (rng.poisson(20 + discount * 200)).astype(int)

    return pd.DataFrame({
        "promotion_id": gen_uuids(N_PROMOTIONS),
        "seller_id": sel["seller_id"].to_numpy(),
        "product_id": sel["product_id"].to_numpy(),
        "discount_pct": discount,
        "start_date": inicio.date if hasattr(inicio, "date") else inicio,
        "end_date": fin,
        "promo_type": promo_type,
        "units_sold_during": units,
    })


# --- 4. warehouse_inventory ------------------------------------------------
def gen_inventory(items: pd.DataFrame, sellers: pd.DataFrame) -> pd.DataFrame:
    """Inventario por producto-vendedor en centros de distribucion."""
    warehouse_ids = gen_uuids(N_WAREHOUSES)
    wh_estado = rng.choice(ESTADOS_BR, size=N_WAREHOUSES)

    pares = items[["product_id", "seller_id"]].drop_duplicates().reset_index(drop=True)
    sel = pares.iloc[rng.choice(len(pares), size=min(N_INVENTORY_PAIRS, len(pares)), replace=False)]
    sel = sel.reset_index(drop=True)
    # 1 o 2 warehouses por par
    n_wh = rng.integers(1, 3, size=len(sel))
    rep_idx = np.repeat(np.arange(len(sel)), n_wh)
    wh_pick = rng.integers(0, N_WAREHOUSES, size=len(rep_idx))

    fecha0 = pd.Timestamp("2018-01-01")
    last_restock = fecha0 + pd.to_timedelta(rng.integers(0, 300, size=len(rep_idx)), unit="D")

    inv = pd.DataFrame({
        "warehouse_id": warehouse_ids[wh_pick],
        "product_id": sel["product_id"].to_numpy()[rep_idx],
        "seller_id": sel["seller_id"].to_numpy()[rep_idx],
        "stock_qty": rng.integers(0, 500, size=len(rep_idx)),
        "reorder_point": rng.integers(10, 100, size=len(rep_idx)),
        "last_restocked_date": last_restock,
    })
    # Un par (producto, vendedor) puede caer 2 veces en el mismo warehouse: deduplicar
    # por la PK compuesta (warehouse_id, product_id, seller_id). Sin RNG adicional aqui,
    # de modo que las tablas generadas despues quedan identicas.
    return inv.drop_duplicates(
        subset=["warehouse_id", "product_id", "seller_id"]).reset_index(drop=True)


# --- 5. order_items extendido (carrier_id + price_event_id) -----------------
def extender_items(
    items: pd.DataFrame, orders: pd.DataFrame, customers: pd.DataFrame,
    price_history: pd.DataFrame, mapa_estado: dict[str, list[str]],
) -> pd.DataFrame:
    """Anade carrier_id (por orden, segun estado) y price_event_id (precio vigente)."""
    # Estado del cliente por orden
    o = pd.merge(orders[["order_id", "customer_id", "order_purchase_timestamp"]],
                 customers, on="customer_id", how="left")
    estados = o["customer_state"].fillna("SP")
    carrier_por_orden = np.empty(len(o), dtype=object)
    todos = [cid for lst in mapa_estado.values() for cid in lst]
    for est, grp_idx in o.groupby(estados).groups.items():
        cids = mapa_estado.get(est, todos)
        pos = o.index.get_indexer(list(grp_idx))
        carrier_por_orden[pos] = rng.choice(cids, size=len(pos))
    o["carrier_id"] = carrier_por_orden

    items_ext = pd.merge(items, o[["order_id", "carrier_id", "order_purchase_timestamp"]],
                         on="order_id", how="left")

    # price_event_id: ultimo evento del par con change_at <= compra (merge_asof)
    left = items_ext.dropna(subset=["order_purchase_timestamp"]).sort_values("order_purchase_timestamp")
    right = price_history.sort_values("change_at")
    asof = pd.merge_asof(
        left, right[["product_id", "seller_id", "change_at", "price_event_id"]],
        left_on="order_purchase_timestamp", right_on="change_at",
        by=["product_id", "seller_id"], direction="backward",
    )
    mapa_pe = asof.set_index(left.index)["price_event_id"]
    items_ext["price_event_id"] = mapa_pe.reindex(items_ext.index)

    cols = ["order_id", "order_item_id", "product_id", "seller_id",
            "shipping_limit_date", "price", "freight_value",
            "price_event_id", "carrier_id"]
    return items_ext[cols]


# --- 6. order_payments extendido (promotion_id) ----------------------------
def extender_payments(
    payments: pd.DataFrame, items: pd.DataFrame, orders: pd.DataFrame,
    promotions: pd.DataFrame,
) -> pd.DataFrame:
    """Asigna promotion_id a pagos cuya orden tuvo un item bajo promocion vigente."""
    comp = pd.merge(
        items[["order_id", "product_id", "seller_id"]],
        promotions[["promotion_id", "product_id", "seller_id", "start_date", "end_date"]],
        on=["product_id", "seller_id"], how="inner",
    )
    comp = pd.merge(comp, orders[["order_id", "order_purchase_timestamp"]],
                    on="order_id", how="left")
    sd = pd.to_datetime(comp["start_date"])
    ed = pd.to_datetime(comp["end_date"])
    vig = (comp["order_purchase_timestamp"] >= sd) & (comp["order_purchase_timestamp"] <= ed)
    comp = comp[vig].drop_duplicates(subset="order_id", keep="first")

    pay = pd.merge(payments, comp[["order_id", "promotion_id"]], on="order_id", how="left")
    cols = ["order_id", "payment_sequential", "payment_type",
            "payment_installments", "payment_value", "promotion_id"]
    return pay[cols]


# --- helpers de senal (review / entrega tardia) ----------------------------
def _orders_con_senal(orders: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Orders entregadas con review_score y flag de entrega tardia."""
    o = orders[orders["order_delivered_customer_date"].notna()].copy()
    o = pd.merge(o, reviews.drop_duplicates("order_id"), on="order_id", how="left")
    o["tardia"] = (o["order_delivered_customer_date"] > o["order_estimated_delivery_date"]).astype(int)
    o["score"] = o["review_score"].fillna(5)
    return o


# --- 7. delivery_incidents -------------------------------------------------
def gen_incidents(orders: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Incidencias de entrega sesgadas a review bajo / entrega tardia."""
    o = _orders_con_senal(orders, reviews).reset_index(drop=True)
    w = 1.0 + o["tardia"].to_numpy() * 3.0 + (o["score"].to_numpy() <= 2) * 4.0
    idx = weighted_sample_idx(w, N_INCIDENTS)
    sub = o.iloc[idx].reset_index(drop=True)

    tipos = rng.choice(
        ["retraso", "producto_danado", "perdida", "direccion_incorrecta", "entrega_fallida"],
        size=len(sub), p=[0.45, 0.2, 0.1, 0.15, 0.1])
    reported = sub["order_delivered_customer_date"] + pd.to_timedelta(
        rng.integers(0, 6, size=len(sub)), unit="D")
    dias_res = rng.integers(1, 31, size=len(sub))
    resolved = reported + pd.to_timedelta(dias_res, unit="D")
    sin_resolver = rng.random(len(sub)) < 0.2
    resolved = resolved.where(~sin_resolver, pd.NaT)

    resolution = rng.choice(
        ["reembolso", "reenvio", "compensacion", "sin_resolucion", "disculpa"],
        size=len(sub), p=[0.25, 0.25, 0.2, 0.15, 0.15])
    comp = np.where(np.isin(resolution, ["reembolso", "compensacion"]),
                    np.round(rng.uniform(10, 300, size=len(sub)), 2), 0.0)

    return pd.DataFrame({
        "incident_id": gen_uuids(len(sub)),
        "order_id": sub["order_id"].to_numpy(),
        "incident_type": tipos,
        "reported_date": reported,
        "resolved_date": resolved,
        "resolution_type": resolution,
        "compensation_value": comp,
    })


# --- 8. customer_support_tickets -------------------------------------------
def gen_tickets(orders: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Tickets de soporte sesgados a review bajo; ligados a orden y cliente."""
    o = pd.merge(orders, reviews.drop_duplicates("order_id"), on="order_id", how="left")
    o = o.reset_index(drop=True)
    score = o["review_score"].fillna(5).to_numpy()
    w = 1.0 + (score <= 2) * 4.0 + (score == 3) * 1.5
    idx = weighted_sample_idx(w, N_TICKETS)
    sub = o.iloc[idx].reset_index(drop=True)

    base = sub["order_delivered_customer_date"].fillna(sub["order_purchase_timestamp"])
    created = base + pd.to_timedelta(rng.integers(0, 10, size=len(sub)), unit="D")
    category = rng.choice(
        ["envio", "producto", "pago", "reembolso", "garantia", "otro"],
        size=len(sub), p=[0.3, 0.25, 0.12, 0.13, 0.1, 0.1])
    priority = rng.choice(["baja", "media", "alta", "critica"],
                          size=len(sub), p=[0.35, 0.4, 0.2, 0.05])
    res_hr = np.round(rng.gamma(shape=2.0, scale=12.0, size=len(sub)), 1)
    sc = sub["review_score"].to_numpy()
    satis = np.where(np.isnan(sc),
                     rng.integers(1, 6, size=len(sub)),
                     np.clip(np.round(np.nan_to_num(sc, nan=3) + rng.integers(-1, 2, size=len(sub))), 1, 5))
    resolved = rng.random(len(sub)) < 0.85

    return pd.DataFrame({
        "ticket_id": gen_uuids(len(sub)),
        "customer_id": sub["customer_id"].to_numpy(),
        "order_id": sub["order_id"].to_numpy(),
        "created_at": created,
        "category": category,
        "priority": priority,
        "resolution_time_hr": res_hr,
        "satisfaction_score": satis.astype(int),
        "resolved": resolved,
    })


# --- 9. product_returns ----------------------------------------------------
def gen_returns(items: pd.DataFrame, orders: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Devoluciones de productos sesgadas a review bajo; ligadas a orden e item."""
    ent = orders[orders["order_delivered_customer_date"].notna()][
        ["order_id", "order_delivered_customer_date"]]
    it = pd.merge(items[["order_id", "product_id", "price"]], ent, on="order_id", how="inner")
    it = pd.merge(it, reviews.drop_duplicates("order_id"), on="order_id", how="left").reset_index(drop=True)
    score = it["review_score"].fillna(5).to_numpy()
    w = 1.0 + (score <= 2) * 4.0
    idx = weighted_sample_idx(w, N_RETURNS)
    sub = it.iloc[idx].reset_index(drop=True)

    reason = rng.choice(
        ["defectuoso", "no_coincide_descripcion", "talla_incorrecta",
         "llego_tarde", "cambio_opinion", "danado_envio"],
        size=len(sub), p=[0.25, 0.2, 0.15, 0.1, 0.15, 0.15])
    return_date = sub["order_delivered_customer_date"] + pd.to_timedelta(
        rng.integers(1, 31, size=len(sub)), unit="D")
    factor = rng.choice([1.0, 0.5], size=len(sub), p=[0.7, 0.3])  # reembolso total o parcial
    refund = np.round(sub["price"].to_numpy() * factor, 2)
    method = rng.choice(["tarjeta", "credito_tienda", "transferencia", "reembolso_original"],
                        size=len(sub), p=[0.4, 0.25, 0.15, 0.2])
    reestocked = rng.random(len(sub)) < 0.6

    return pd.DataFrame({
        "return_id": gen_uuids(len(sub)),
        "order_id": sub["order_id"].to_numpy(),
        "product_id": sub["product_id"].to_numpy(),
        "return_reason": reason,
        "return_date": return_date,
        "refund_amount": refund,
        "refund_method": method,
        "reestocked": reestocked,
    })


# --- Validacion ------------------------------------------------------------
def validar(tablas: dict[str, pd.DataFrame], orig: dict[str, pd.DataFrame]) -> None:
    """Asserts de integridad referencial y de rangos."""
    order_ids = set(orig["orders"]["order_id"])
    product_ids = set(orig["items"]["product_id"])
    seller_ids = set(orig["items"]["seller_id"])
    customer_ids = set(orig["customers"]["customer_id"])
    carrier_ids = set(tablas["carriers"]["carrier_id"])
    price_ids = set(tablas["product_price_history"]["price_event_id"])
    promo_ids = set(tablas["seller_promotions"]["promotion_id"])

    def subset(col: pd.Series, universo: set, nombre: str) -> None:
        vals = set(col.dropna().astype(str))
        assert vals <= universo, f"FK fuera de universo en {nombre}: {list(vals - universo)[:3]}"

    subset(tablas["delivery_incidents"]["order_id"], order_ids, "delivery_incidents.order_id")
    subset(tablas["customer_support_tickets"]["order_id"], order_ids, "tickets.order_id")
    subset(tablas["customer_support_tickets"]["customer_id"], customer_ids, "tickets.customer_id")
    subset(tablas["product_returns"]["order_id"], order_ids, "returns.order_id")
    subset(tablas["product_returns"]["product_id"], product_ids, "returns.product_id")
    subset(tablas["product_price_history"]["product_id"], product_ids, "price_history.product_id")
    subset(tablas["product_price_history"]["seller_id"], seller_ids, "price_history.seller_id")
    subset(tablas["seller_promotions"]["product_id"], product_ids, "promotions.product_id")
    subset(tablas["seller_promotions"]["seller_id"], seller_ids, "promotions.seller_id")
    subset(tablas["warehouse_inventory"]["product_id"], product_ids, "inventory.product_id")
    subset(tablas["warehouse_inventory"]["seller_id"], seller_ids, "inventory.seller_id")
    subset(tablas["olist_order_items_extended"]["carrier_id"], carrier_ids, "items.carrier_id")
    subset(tablas["olist_order_items_extended"]["price_event_id"], price_ids, "items.price_event_id")
    subset(tablas["olist_order_payments_extended"]["promotion_id"], promo_ids, "payments.promotion_id")

    # Rangos
    c = tablas["carriers"]
    assert c["on_time_rate"].between(0, 1).all(), "on_time_rate fuera de [0,1]"
    assert tablas["seller_promotions"]["discount_pct"].between(0, 1).all(), "discount fuera de [0,1]"
    assert tablas["customer_support_tickets"]["satisfaction_score"].between(1, 5).all(), "satisfaction fuera de [1,5]"

    # Unicidad de PK compuestas
    inv = tablas["warehouse_inventory"]
    assert not inv.duplicated(subset=["warehouse_id", "product_id", "seller_id"]).any(), \
        "PK compuesta duplicada en warehouse_inventory"
    print("OK validaciones de integridad y rangos.")


# --- Orquestacion ----------------------------------------------------------
def main() -> None:
    """Genera y persiste todas las tablas sinteticas y extendidas."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    orig = cargar_originales()

    carriers, mapa_estado = gen_carriers()
    price_history = gen_price_history(orig["items"])
    promotions = gen_promotions(orig["items"])
    inventory = gen_inventory(orig["items"], orig["sellers"])
    items_ext = extender_items(orig["items"], orig["orders"], orig["customers"],
                               price_history, mapa_estado)
    payments_ext = extender_payments(orig["payments"], orig["items"], orig["orders"], promotions)
    incidents = gen_incidents(orig["orders"], orig["reviews"])
    tickets = gen_tickets(orig["orders"], orig["reviews"])
    returns = gen_returns(orig["items"], orig["orders"], orig["reviews"])

    tablas = {
        "carriers": carriers,
        "product_price_history": price_history,
        "seller_promotions": promotions,
        "warehouse_inventory": inventory,
        "delivery_incidents": incidents,
        "customer_support_tickets": tickets,
        "product_returns": returns,
        "olist_order_items_extended": items_ext,
        "olist_order_payments_extended": payments_ext,
    }

    validar(tablas, orig)

    for nombre, df in tablas.items():
        ruta = OUT_DIR / f"{nombre}.csv"
        df.to_csv(ruta, index=False)
        print(f"  {nombre:32s} {len(df):>8,} filas -> {ruta.name}")

    print(f"\nListo. {len(tablas)} CSV escritos en {OUT_DIR}")


if __name__ == "__main__":
    main()
