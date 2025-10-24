import io
import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF

st.set_page_config(page_title="Detector de Manchetes (PDF Clipping ALMG)", layout="wide")

st.title("📰 Detector de Manchetes em PDF (Clipping ALMG)")
st.write(
    "Faça upload do PDF de clipping e o app tentará identificar as **manchetes** com base em tamanho de fonte, negrito e posição na página."
)

uploaded = st.file_uploader("Envie o arquivo PDF", type=["pdf"])

# ------------------------------
# Heurísticas configuráveis
# ------------------------------
st.sidebar.header("Ajustes das heurísticas")
perc_font_threshold = st.sidebar.slider(
    "Percentil mínimo do tamanho de fonte (recomendado: 85–95)",
    min_value=50, max_value=99, value=90, step=1
)
min_len = st.sidebar.slider(
    "Tamanho mínimo do texto (caracteres)",
    min_value=4, max_value=50, value=6, step=1
)
max_len = st.sidebar.slider(
    "Tamanho máximo do texto (caracteres)",
    min_value=40, max_value=200, value=120, step=5
)
upper_ratio_threshold = st.sidebar.slider(
    "Proporção mínima de MAIÚSCULAS (0–1)",
    min_value=0.0, max_value=1.0, value=0.4, step=0.05
)
prefer_bold = st.sidebar.checkbox("Dar prioridade a textos em **negrito**", value=True)
exclude_footers = st.sidebar.checkbox("Tentar ignorar cabeçalhos/rodapés repetidos", value=True)

def is_bold(span_flags: int) -> bool:
    # PyMuPDF: bold costuma ser o bit 2 (valor 2) em muitos casos
    return bool(span_flags & 2)

def upper_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    upp = sum(1 for c in letters if c.isupper())
    return upp / len(letters)

def normalize_text(t: str) -> str:
    return " ".join(t.replace("\n", " ").split())

def extract_spans(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    records = []
    for pno in range(len(doc)):
        page = doc[pno]
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type") != 0:  # 0 = texto
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = normalize_text(span.get("text",""))
                    if not text:
                        continue
                    size = float(span.get("size", 0))
                    flags = int(span.get("flags", 0))
                    bbox = span.get("bbox", [None,None,None,None])
                    y0 = bbox[1] if bbox and len(bbox) >= 2 else None
                    records.append({
                        "page": pno+1,
                        "text": text,
                        "font_size": size,
                        "flags": flags,
                        "is_bold": is_bold(flags),
                        "y0": y0,
                    })
    return pd.DataFrame.from_records(records)

def filter_headlines(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    cutoff = np.percentile(df["font_size"], perc_font_threshold)
    candidates = df.copy()
    candidates["upper_ratio"] = candidates["text"].apply(upper_ratio)
    candidates["length"] = candidates["text"].str.len()

    base = (
        (candidates["font_size"] >= cutoff) |
        ((candidates["is_bold"]) & (candidates["font_size"] >= df["font_size"].median()))
    )

    content = (
        (candidates["length"] >= min_len) &
        (candidates["length"] <= max_len) &
        (candidates["upper_ratio"] >= upper_ratio_threshold)
    )

    mask = base & content
    out = candidates.loc[mask].copy()

    # Remover textos repetidos (cabeçalhos/rodapés)
    if exclude_footers and not out.empty:
        counts = out["text"].value_counts()
        repetitive = set(counts[counts >= max(3, int(len(df)/10))].index)
        out = out[~out["text"].isin(repetitive)]

    # Prioridade
    if prefer_bold and not out.empty:
        out["priority"] = out["is_bold"].astype(int) * 2 + (out["font_size"] >= cutoff).astype(int)
    else:
        out["priority"] = (out["font_size"] >= cutoff).astype(int)

    # Ordenação
    out = out.sort_values(by=["page", "priority", "y0", "font_size"], ascending=[True, False, True, False])

    # Deduplicação simples por proximidade vertical e texto semelhante
    dedup = []
    seen_by_page = {}
    for _, row in out.iterrows():
        page = row["page"]
        y = row["y0"] if pd.notnull(row["y0"]) else 1e9
        tnorm = row["text"].casefold()
        key = (page, tnorm)
        if key in seen_by_page:
            continue
        close = False
        for (py, ty, yy) in seen_by_page.get(page, []):
            if abs(yy - y) <= 3 and (tnorm == ty or tnorm in ty or ty in tnorm):
                close = True
                break
        if not close:
            dedup.append(row)
            seen_by_page.setdefault(page, []).append((page, tnorm, y))

    out = pd.DataFrame(dedup)

    return out[["page", "text", "font_size", "is_bold", "upper_ratio", "y0"]].rename(columns={"text": "manchete"})

if uploaded:
    pdf_bytes = uploaded.read()
    with st.spinner("Analisando o PDF..."):
        spans_df = extract_spans(pdf_bytes)
        if spans_df.empty:
            st.error("Não encontrei texto embutido no PDF. Se o arquivo for **escaneado** (imagem), será necessário aplicar OCR antes.")
        else:
            heads = filter_headlines(spans_df)
            st.subheader("Resultados")
            st.write(f"Spans de texto encontrados: **{len(spans_df):,}**")
            st.write(f"Manchetes detectadas: **{len(heads):,}**")

            if not heads.empty:
                st.dataframe(heads, use_container_width=True)
                csv = heads.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Baixar CSV de manchetes",
                    data=csv,
                    file_name="manchetes_detectadas.csv",
                    mime="text/csv"
                )
            else:
                st.info("Nenhuma manchete detectada com as heurísticas atuais. Tente ajustar os controles na barra lateral (por exemplo, reduzir o percentil de fonte, diminuir o limite de maiúsculas, etc.).")

    with st.expander("ℹ️ Dicas rápidas"):
        st.markdown(
            """
            - Se o PDF for **digital** (não-escaneado), este método costuma funcionar bem.
            - Para arquivos **escaneados**, será necessário aplicar OCR (Tesseract, Google Document AI, ABBYY, etc.) antes de usar este app.
            - Ajuste o *percentil de fonte* para ser mais ou menos rigoroso (85–95% geralmente vai bem).
            - Ative "**ignorar cabeçalhos/rodapés**" para reduzir falsos positivos recorrentes.
            - A proporção de MAIÚSCULAS ajuda a filtrar subtítulos e corpo de texto.
            """
        )
else:
    st.info("Envie um PDF de clipping para começar.")
