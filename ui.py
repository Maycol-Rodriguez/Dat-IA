"""Interfaz Streamlit para consultar la API Dat-IA."""

import json
import os
from typing import Any
from urllib import error, request

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.set_page_config(page_title="📊 Asistente DatIA", layout="wide")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _ensure_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "Puedes consultar el esquema de la base de datos y generar SQL."}
        ]
    if "tool_logs" not in st.session_state:
        st.session_state.tool_logs = []
    if "initialized" not in st.session_state:
        st.session_state.initialized = True


def _api_call(path: str, *, method: str = "GET", payload: dict[str, Any] | None = None, files: dict[str, tuple[Any, ...]] | None = None) -> tuple[dict[str, Any] | None, str | None]:
    url = f"{API_BASE_URL}{path}"
    headers: dict[str, str] = {}
    data: bytes | None = None

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif files is not None:
        boundary = "----DatIAStreamlitBoundary"
        parts = []
        for field_name, (filename, content, content_type) in files.items():
            parts.append(f"--{boundary}\r\n".encode("utf-8"))
            parts.append(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8"))
            parts.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
            parts.append(content if isinstance(content, bytes) else str(content).encode("utf-8"))
            parts.append(b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        data = b"".join(parts)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

    req = request.Request(url, data=data, headers=headers, method=method)

    try:
        with request.urlopen(req, timeout=120) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}, None
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore")
        try:
            detail = json.loads(raw)
        except json.JSONDecodeError:
            detail = raw or str(exc)
        return None, str(detail)
    except error.URLError as exc:
        return None, str(exc.reason)


def _render_messages() -> None:
    for msg in st.session_state.messages:
        if msg.get("role") in ("system", "tool"):
            continue
        if msg.get("role") == "assistant" and not msg.get("content"):
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def _render_tool_logs() -> None:
    st.header("🔧 Llamadas a Herramientas")
    if not st.session_state.tool_logs:
        st.caption("Aún no hay trazas de herramientas para esta sesión.")
        return

    for log in st.session_state.tool_logs:
        with st.expander(f"🔨 {log.get('name', 'tool')}", expanded=False):
            st.subheader("Argumentos")
            st.code(json.dumps(log.get("arguments", {}), indent=2, ensure_ascii=False), language="json")
            st.subheader("Resultado")
            truncated = json.dumps(log.get("result", {}), indent=2, ensure_ascii=False)
            if log.get("is_error"):
                st.warning(truncated)
            else:
                st.code(truncated, language="json")


_ensure_state()

st.title("📊 Asistente DatIA")
st.caption("Consulta de análisis de datos de forma conversacional.")

_render_messages()

if user_input := st.chat_input("Escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Procesando..."):
            response, error = _api_call("/query/json", method="POST", payload={"question": user_input})

        if error:
            message = f"Error al consultar la API: {error}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
            st.session_state.tool_logs = []
        else:
            sql = response.get("sql", "") or ""
            sources = response.get("sources", "") or ""
            confidence_note = response.get("confidence_note", "") or ""
            if response.get("tool_logs"):
                st.session_state.tool_logs = response["tool_logs"]

            assistant_text = f"**SQL generado**\n```sql\n{sql}\n```"
            if sources:
                assistant_text += f"\n\n**Fuentes:** {sources}"
            if confidence_note:
                assistant_text += f"\n\n**Nota de confianza:** {confidence_note}"

            st.markdown(assistant_text)
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})

with st.sidebar:
    st.header("📥 Indexar esquema")
    uploaded_file = st.file_uploader("Subir JSON con tablas y DDL", type=["json"])
    if st.button("Indexar esquema") and uploaded_file is not None:
        content = uploaded_file.read()
        response, error = _api_call(
            "/ingest",
            method="POST",
            files={"file": (uploaded_file.name, content, "application/json")},
        )
        if error:
            st.error(f"No se pudo indexar: {error}")
        else:
            st.success(f"Se indexaron {response.get('chunks_indexed', 0)} tablas.")
            if response.get("tool_logs"):
                st.session_state.tool_logs = response["tool_logs"]

    st.divider()
    _render_tool_logs()
