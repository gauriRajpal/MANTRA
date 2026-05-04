import os
import streamlit as st
import pandas as pd
import sentencepiece as spm

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.bin.translate import translate as onmt_translate

from lingua import Language, LanguageDetectorBuilder



# Training models here

MODEL_REGISTRY = {
    ("Hindi", "Punjabi"): {
        "model_path": "models/model.hi_pun_step_3000.pt",
        "src_spm": "hindi_punjabi/source.model",
        "tgt_spm": "hindi_punjabi/target.model",
        "beam_size": 5
    },
    ("Hindi","Bengali"):{
        "model_path":"models/model.hi_bn_step_6000.pt",
        "src_spm":"hindi_bengali/source.model",
        "tgt_spm":"hindi_bengali/target.model",
        "beam_size":5
    },
    ("Hindi","Bhojpuri"):{
        "model_path":"models/model.hi_bho_step_5000.pt",
        "src_spm":"hindi_bhojpuri/source.model",
        "tgt_spm":"hindi_bhojpuri/target.model",
        "beam_size":5
    },
    ("Hindi","Sindhi"):{
        "model_path":"models/model.hi_sn_step_15000.pt",
        "src_spm":"hindi_sindhi/source.model",
        "tgt_spm":"hindi_sindhi/target.model",
        "beam_size":5
    },
    ("Punjabi","Hindi"):{
        "model_path":"models/model.pun_hi_step_3000.pt",
        "src_spm" : "punjabi_hindi/source.model",
        "tgt_spm":"punjabi_hindi/target.model",
        "beam_size":5
    },
    ("Punjabi","Bengali"):{
        "model_path":"models/model.pun_bn_step_3000.pt",
        "src_spm":"punjabi_bengali/source.model",
        "tgt_spm":"punjabi_bengali/target.model",
        "beam_size":5
    },
    ("Punjabi","Bhojpuri"):{
        "model_path":"models/model.pun_bho_step_5000.pt",
        "src_spm":"punjabi_bhojpuri/source.model",
        "tgt_spm":"punjabi_bhojpuri/target.model",
        "beam_size":5
    },
    ("Punjabi","Sindhi"):{
        "model_path":"models/model.pun_sn_step_1000.pt",
        "src_spm":"punjabi_sindhi/source.model",
        "tgt_spm":"punjabi_sindhi/target.model",
        "beam_size":5
    },
    ("Bengali","Hindi"):{
        "model_path":"models/model.bn_hi_step_14000.pt",
        "src_spm":"bengali_hindi/source.model",
        "tgt_spm":"bengali_hindi/target.model",
        "beam_size":5
    },
    ("Bengali","Punjabi"):{
        "model_path":"models/model.bn_pun_step_15000.pt",
        "src_spm":"bengali_punjabi/source.model",
        "tgt_spm":"bengali_punjabi/target.model",
        "beam_size":5
    },
    ("Bengali","Bhojpuri"):{
        "model_path":"models/model.bn_bho_step_6000.pt",
        "src_spm":"bengali_bhojpuri/source.model",
        "tgt_spm":"bengali_bhojpuri/target.model",
        "beam_size":5
    },
    ("Bengali","Sindhi"):{
        "model_path":"models/model.bn_sn_step_1000.pt",
        "src_spm":"bengali_sindhi/source.model",
        "tgt_spm":"bengali_sindhi/target.model",
        "beam_size":5
    },
    ("Bhojpuri","Hindi"):{
        "model_path":"models/model.bho_hi_step_1000.pt",
        "src_spm":"bhojpuri_hindi/source.model",
        "tgt_spm":"bhojpuri_hindi/target.model",
        "beam_size":5
    },
    ("Bhojpuri","Bengali"):{
        "model_path":"models/model.bho_bn_step_6000.pt",
        "src_spm":"bhojpuri_bengali/source.model",
        "tgt_spm":"bhojpuri_bengali/target.model",
        "beam_size":5
    },
    ("Bhojpuri","Punjabi"):{
        "model_path":"models/model.bho_pun_step_5000.pt",
        "src_spm":"bhojpuri_punjabi/source.model",
        "tgt_spm":"bhojpuri_punjabi/target.model",
        "beam_size":5
    },
    ("Bhojpuri","Sindhi"):{
        "model_path":"models/model.bho_sn_step_1000.pt",
        "src_spm":"bhojpuri_sindhi/source.model",
        "tgt_spm":"bhojpuri_sindhi/target.model",
        "beam_size":5
    },
    ("Sindhi","Hindi"):{
        "model_path":"models/model.sn_hi_step_3000.pt",
        "src_spm":"sindhi_hindi/source.model",
        "tgt_spm":"sindhi_hindi/target.model",
        "beam_size":5
    },
    ("Sindhi","Bhojpuri"):{
        "model_path":"models/model.sn_bho_step_1000.pt",
        "src_spm":"sindhi_bhojpuri/source.model",
        "tgt_spm":"sindhi_bhojpuri/target.model",
        "beam_size":5
    },
    ("Sindhi","Bengali"):{
        "model_path":"models/model.sn_bn_step_1000.pt",
        "src_spm":"sindhi_bengali/source.model",
        "tgt_spm":"sindhi_bengali/target.model",
        "beam_size":5
    },
    ("Sindhi","Punjabi"):{
        "model_path":"models/model.sn_pun_step_1000.pt",
        "src_spm":"sindhi_punjabi/source.model",
        "tgt_spm":"sindhi_punjabi/target.model",
        "beam_size":5
    }
}



#  BACKEND: OpenNMT v3 translation pipeline

@st.cache_resource(show_spinner=False)
def load_spm(spm_path: str):
    sp = spm.SentencePieceProcessor()
    sp.load(spm_path)
    return sp


@st.cache_resource(show_spinner=False)
def build_translate_opt(model_path: str, beam_size: int):
    parser = ArgumentParser()
    opts.translate_opts(parser)

    opt = parser.parse_args([
        "-model", model_path,
        "-src", "temp_src.txt",
        "-output", "temp_pred.txt",
        "-beam_size", str(beam_size),
        "-n_best", "1",
        "-gpu", "-1",  # CPU
        "-replace_unk"
    ])
    return opt


def sp_encode_lines(sp_processor: spm.SentencePieceProcessor, lines: list[str]) -> list[str]:
    """Encode text -> sentencepiece tokens (space separated)."""
    out = []
    for s in lines:
        s = s.strip()
        if not s:
            out.append("")
            continue
        pieces = sp_processor.encode_as_pieces(s)
        out.append(" ".join(pieces))
    return out


def sp_decode_lines(sp_processor: spm.SentencePieceProcessor, token_lines: list[str]) -> list[str]:
    """Decode sentencepiece tokens -> normal text."""
    out = []
    for s in token_lines:
        s = s.strip()
        if not s:
            out.append("")
            continue
        pieces = s.split()
        out.append(sp_processor.decode_pieces(pieces))
    return out


def translate_text(src_text: str, src_lang: str, tgt_lang: str):
    """
    Returns:
        translations (list[str]), sources (list[str])
    """
    key = (src_lang, tgt_lang)
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Model not available for {src_lang} → {tgt_lang}")

    cfg = MODEL_REGISTRY[key]
    model_path = cfg["model_path"]
    src_spm_path = cfg["src_spm"]
    tgt_spm_path = cfg["tgt_spm"]
    beam_size = int(cfg.get("beam_size", 5))

    for p in [model_path, src_spm_path, tgt_spm_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    sp_src = load_spm(src_spm_path)
    sp_tgt = load_spm(tgt_spm_path)
    opt = build_translate_opt(model_path, beam_size)

    sources = [ln.strip() for ln in src_text.splitlines() if ln.strip()]
    if not sources:
        return [], []

    tokenized = sp_encode_lines(sp_src, sources)

    #  input file
    with open(opt.src, "w", encoding="utf-8") as f:
        for line in tokenized:
            f.write(line + "\n")

    # Run OpenNMT Translate
    onmt_translate(opt)

    # Read output tokens
    with open(opt.output, "r", encoding="utf-8") as f:
        preds_token = [x.strip() for x in f.readlines()]

    # Decode using TARGET SPM
    preds = sp_decode_lines(sp_tgt, preds_token)
    return preds, sources



#  LANGUAGE DETECTION

LINGUA_MAP = {
    "Hindi": Language.HINDI,
    "Bengali": Language.BENGALI,
    "Punjabi": Language.PUNJABI,
    "English": Language.ENGLISH,
    
}

@st.cache_resource(show_spinner=False)
def get_detector():
    supported = list(LINGUA_MAP.values())
    return LanguageDetectorBuilder.from_languages(*supported).build()


def check_language_mismatch(text: str, selected_src_lang: str):
    """
    Returns: (detected_language_name, mismatch_bool)
    """
    expected = LINGUA_MAP.get(selected_src_lang)
    if expected is None:
        return None, False  

    detector = get_detector()
    detected = detector.detect_language_of(text)

    if detected is None:
        return None, False

    return detected.name.title(), detected != expected



#  FRONTEND UI

st.set_page_config(page_title="Machine Translation", layout="centered")
st.title("Multilingual Machine Translation System ")
st.caption("")



st.markdown(
    """
    <style>
    textarea[disabled] {
        -webkit-text-fill-color: #ffffff !important;
        color: #ffffff !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


languages = {
    "Bengali": "ben",
    "Hindi": "hi",
    "Punjabi": "pun",
    "Bhojpuri": "bh",
    "Sindhi":"sn"
}


# Session state init
if "source" not in st.session_state:
    st.session_state.source = "Hindi"
if "target" not in st.session_state:
    st.session_state.target = "Punjabi"
if "result" not in st.session_state:
    st.session_state.result = ""
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["source", "translation"])


# Language selection UI
c1, c2, c3 = st.columns([4, 1, 4])

with c1:
    src_lang = st.selectbox(
        "From",
        list(languages.keys()),
        index=list(languages.keys()).index(st.session_state.source)
    )

with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔁"):
        st.session_state.source, st.session_state.target = (
            st.session_state.target,
            st.session_state.source
        )
        st.rerun()

with c3:
    tgt_lang = st.selectbox(
        "To",
        list(languages.keys()),
        index=list(languages.keys()).index(st.session_state.target)
    )

st.session_state.source = src_lang
st.session_state.target = tgt_lang


# Text areas
left, right = st.columns(2)

with left:
    input_text = st.text_area("Enter text", height=200)

with right:
    st.text_area("Translated text", height=200, value=st.session_state.result, disabled=True)


# Language mismatch override checkbox
translate_anyway = st.checkbox("Translate anyway (ignore language mismatch)")


# Translate button
if st.button("Translate", use_container_width=True):
    if input_text.strip() == "":
        st.warning("Enter some text")
    else:
        try:
            #  Check language mismatch
            detected_name, mismatch = check_language_mismatch(input_text, src_lang)

            if mismatch and not translate_anyway:
                st.error(
                    f"❌ Language mismatch!\n\n"
                    f"Selected source: **{src_lang}**\n"
                    f"Detected input language: **{detected_name}**\n\n"
                    f"Either change the source language or enable **Translate anyway**."
                )
                st.stop()

            with st.spinner(f"Translating {src_lang} → {tgt_lang} ..."):
                preds, sources = translate_text(input_text, src_lang, tgt_lang)

            st.session_state.result = "\n".join(preds)

            df = pd.DataFrame({"source": sources, "translation": preds})
            st.session_state.df = df

            st.rerun()

        except ValueError as e:
            st.error(str(e))
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.exception(e)



#  Results Table + Fullscreen + Downloads

if not st.session_state.df.empty:
    st.markdown("### Results (Table View)")

    # Streamlit dataframe has fullscreen icon automatically
    st.dataframe(st.session_state.df, use_container_width=True)

    colA, colB = st.columns(2)

    # CSV Download
    with colA:
        st.download_button(
            "⬇️ Download CSV",
            data=st.session_state.df.to_csv(index=False).encode("utf-8"),
            file_name="translations.csv",
            mime="text/csv",
            use_container_width=True
        )

    # TXT Download
    with colB:
        txt_data = st.session_state.result.strip()
        st.download_button(
            "⬇️ Download TXT",
            data=txt_data.encode("utf-8"),
            file_name="translation.txt",
            mime="text/plain",
            use_container_width=True
        )
