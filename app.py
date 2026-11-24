"""CACTNet morphology classification | Streamlit Community Cloud."""
from __future__ import annotations

import io
import hashlib
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
import altair as alt
import numpy as np
from ftir_matching import ReferenceLibrary, FULL_RANGE, ADAPTIVE_RANGE

from inference import CACTNetPredictor

st.set_page_config(
    page_title="CACTNet | Microplastic Identification System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container {max-width: 1450px; padding-top: 1.1rem; padding-bottom: 2rem;}
    h1 {font-size: 2rem !important; letter-spacing: -0.5px;}
    div[data-testid="stTabs"] button[aria-selected="true"] {color: #ec6d18;}
    div.stButton > button[kind="primary"] {background: #ee741a; border-color: #ee741a;}
    div.stButton > button[kind="primary"]:hover {background: #d95f0b; border-color: #d95f0b;}
    .result-header {font-size:1.25rem; font-weight:700; padding-bottom:0.4rem;}
    .muted-note {font-size:0.85rem; color:#6b7280;}
</style>
""", unsafe_allow_html=True)

MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_IMAGE_PIXELS = 60_000_000
MAX_BATCH_BYTES = 150 * 1024 * 1024
MAX_BATCH = 20
MAX_FTIR_BATCH = 100
MAX_FTIR_BATCH_BYTES = 50 * 1024 * 1024
CLASSES = ["fiber", "film", "fragment", "particle"]


def cloud_setting(key: str) -> str:
    try:
        return str(st.secrets.get(key, os.getenv(key, "")))
    except Exception:
        return os.getenv(key, "")


@st.cache_resource(show_spinner="Loading CACTNet model...")
def get_predictor(weights_url: str, expected_sha256: str, github_token: str) -> CACTNetPredictor:
    return CACTNetPredictor(weights_url=weights_url, expected_sha256=expected_sha256, github_token=github_token)


def open_uploaded(uploaded) -> Image.Image:
    if uploaded.size > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 100 MiB upload limit.")
    with Image.open(io.BytesIO(uploaded.getvalue())) as src:
        pixels = src.width * src.height
        if pixels > MAX_IMAGE_PIXELS:
            raise ValueError("Image exceeds the 60-megapixel safety limit. Please resize it externally.")
        if pixels > 24_000_000:
            src.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
        return src.convert("RGB")


def show_preview(img):
    preview = img.copy()
    preview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    st.image(preview, use_container_width=True)


def upload_identity(uploaded):
    return (uploaded.name, uploaded.size, hashlib.sha256(uploaded.getvalue()).hexdigest())


def prepare_model():
    url = cloud_setting("CACTNET_WEIGHTS_URL")
    digest = cloud_setting("CACTNET_WEIGHTS_SHA256")
    github_token = cloud_setting("CACTNET_GITHUB_TOKEN")
    try:
        return get_predictor(url, digest, github_token)
    except Exception as exc:
        st.error(f"CACTNet model is unavailable: {exc}")
        return None


def show_probabilities(pred):
    st.markdown("**Class Probabilities**")
    for label, value in pred.probabilities.items():
        a, b = st.columns([1.1, 4.2], vertical_alignment="center")
        with a:
            st.write(label.capitalize())
        with b:
            st.progress(min(max(value, 0.0), 1.0), text=f"{value * 100:.2f}%")


def show_single():
    col_img, col_result = st.columns([1.06, 0.94], gap="large")
    with col_img:
        uploaded = st.file_uploader(
            "Microscopy Image", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"],
            key="single_uploader",
        )
        st.info(
            "**Image requirements:** This system is designed for morphology classification "
            "of microplastic microscopy images. Please upload images containing microplastics. "
            "Non-microplastic images may produce unreliable predictions.",
            icon="ℹ️",
        )
        if uploaded is not None:
            try:
                img = open_uploaded(uploaded)
            except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                st.error(f"Unable to open the image: {exc}")
                return
            show_preview(img)
        run = st.button("Classify Image", type="primary", use_container_width=True, key="run_single")
    with col_result:
        st.markdown('<div class="result-header">Classification Result</div>', unsafe_allow_html=True)
        if run:
            if uploaded is None:
                st.warning("Please upload a microscopy image.")
            else:
                predictor = prepare_model()
                if predictor is not None:
                    try:
                        with st.spinner("Classifying image..."):
                            pred = predictor.predict(img)
                        st.session_state["single_result"] = (upload_identity(uploaded), pred)
                    except Exception as exc:
                        st.error(f"Classification failed: {exc}")
        cached = st.session_state.get("single_result")
        if cached and uploaded is not None and cached[0] == upload_identity(uploaded):
            pred = cached[1]
            st.success(f"Predicted class: **{pred.label.capitalize()}**")
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{pred.confidence * 100:.2f}%")
            m2.metric("Model", "CACTNet")
            show_probabilities(pred)
        else:
            st.info("Upload an image and select Classify Image.")


def classify_batch(uploaded_files):
    if len(uploaded_files) > MAX_BATCH:
        st.warning(f"Please select no more than {MAX_BATCH} images per batch.")
        return
    if sum(f.size for f in uploaded_files) > MAX_BATCH_BYTES:
        st.warning("The total batch size must not exceed 150 MiB. Please use smaller batches.")
        return
    predictor = prepare_model()
    if predictor is None:
        return
    rows = []
    prog = st.progress(0, text="Classifying images...")
    for i, uploaded in enumerate(uploaded_files):
        row = {"image_name": Path(uploaded.name).name}
        try:
            img = open_uploaded(uploaded)
            pred = predictor.predict(img)
            row.update({
                "pred_label": pred.label,
                "pred_confidence": round(pred.confidence, 6),
                **{f"prob_{name}": round(pred.probabilities[name], 6) for name in CLASSES},
                "error": "",
            })
        except Exception as exc:
            row.update({
                "pred_label": "", "pred_confidence": None,
                **{f"prob_{name}": None for name in CLASSES},
                "error": str(exc),
            })
        rows.append(row)
        prog.progress((i + 1) / len(uploaded_files), text=f"Processed {i + 1}/{len(uploaded_files)}")
    prog.empty()
    return pd.DataFrame(rows)


def show_batch():
    uploaded_files = st.file_uploader(
        "Upload microscopy images for batch morphology classification",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"],
        key="batch_uploader",
    )
    st.info(
        "**Image requirements:** This system is designed for morphology classification "
        "of microplastic microscopy images. Please upload images containing microplastics. "
        "Non-microplastic images may produce unreliable predictions.",
        icon="ℹ️",
    )
    if st.button("Classify Images", type="primary", use_container_width=True, key="run_batch"):
        if not uploaded_files:
            st.warning("Please upload microscopy images.")
        else:
            frame = classify_batch(uploaded_files)
            if frame is not None:
                st.session_state["batch_result"] = (tuple(upload_identity(f) for f in uploaded_files), frame)
    result = st.session_state.get("batch_result")
    file_id = tuple(upload_identity(f) for f in uploaded_files) if uploaded_files else ()
    if result and file_id == result[0]:
        frame = result[1]
        success = frame["error"].eq("")
        st.subheader("Classification Results")
        st.write(f"Classification complete: **{int(success.sum())} succeeded, {int((~success).sum())} failed**")
        successes = frame[success]
        st.caption("Image Predictions")
        st.dataframe(frame, hide_index=True, use_container_width=True)
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.caption("Predicted Class Counts")
            counts = Counter(successes["pred_label"])
            summary = pd.DataFrame({
                "Class": [s.capitalize() for s in CLASSES],
                "Predicted images": [counts.get(s, 0) for s in CLASSES],
            })
            st.dataframe(summary, hide_index=True, use_container_width=True)
        with right:
            st.caption("Export")
            st.download_button(
                "Download Prediction CSV",
                data=frame.to_csv(index=False).encode("utf-8-sig"),
                file_name="CACTNet_batch_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )


@st.cache_resource(show_spinner="Loading FTIR reference library...")
def get_library():
    return ReferenceLibrary()


def matching_mode(prefix):
    mode = st.radio('Matching mode', [FULL_RANGE, ADAPTIVE_RANGE],
                    horizontal=True, key=prefix+'_mode')
    if mode == FULL_RANGE:
        st.caption('Required range: 650–4000 cm⁻¹; maximum gap: 10 cm⁻¹.')
    else:
        st.caption('Uses the common measured range only (at least 70% of 650–4000 cm⁻¹). Maximum gap within that range: 10 cm⁻¹. Retain the fingerprint region where possible.')
    return mode


def match_metadata(result):
    return {k: result[k] for k in ('mode', 'range_min_cm1', 'range_max_cm1', 'coverage_percent', 'fingerprint_complete')}


def spectral_inputs(prefix):
    uploaded = st.file_uploader("FTIR Spectrum (SP, CSV, TXT)", type=["sp", "csv", "txt"], key=prefix+"_sp")
    signal = "Absorbance"
    if uploaded is not None and Path(uploaded.name).suffix.lower() != ".sp":
        signal = st.selectbox("Signal type", ["Absorbance", "Transmittance (%)", "Transmittance (fraction)"], key=prefix+"_unit")
    st.caption("PerkinElmer binary SP: units are read automatically. CSV/TXT: two numeric columns, wavenumber in cm⁻¹ and signal. ")
    return uploaded, signal, matching_mode(prefix)


POLYMER_NAMES = {
    'ABS': 'Acrylonitrile butadiene styrene', 'AC': 'Acrylic',
    'CA': 'Cellulose acetate', 'EAA': 'Ethylene acrylic acid copolymer',
    'PA': 'Polyamide', 'PC': 'Polycarbonate', 'PE': 'Polyethylene',
    'PET': 'Polyethylene terephthalate', 'PEVA': 'Ethylene-vinyl acetate copolymer',
    'PLA': 'Polylactic acid', 'PMMA': 'Polymethyl methacrylate',
    'PP': 'Polypropylene', 'PS': 'Polystyrene', 'PU': 'Polyurethane',
    'PVC': 'Polyvinyl chloride',
}
SPECTRUM_COLORS = ['#0072B2', '#D55E00']  # Sample, Reference
def wavenumber_ticks(low, high):
    ticks = [float(high)]
    for tick in range(4000, 499, -500):
        if low + 120 <= tick <= high - 120:
            ticks.append(float(tick))
    ticks.append(float(low))
    return sorted(set(ticks), reverse=True)


def polymer_name(code):
    code = str(code)
    return f'{POLYMER_NAMES[code]} ({code})' if code in POLYMER_NAMES else code


def spectrum_frame(result):
    def zscore(y):
        y = np.asarray(y, dtype=float)
        return (y - y.mean()) / y.std()
    return pd.DataFrame({'Wavenumber': result['axis'],
                         'Sample': zscore(result['query']),
                         'Reference': zscore(result['reference'])})


def spectral_svg(frame):
    """Export the same standardized curves as a standalone vector image."""
    w, h = 1000, 560
    left, right, top, bottom = 96, 965, 66, 478
    ymin = float(min(frame['Sample'].min(), frame['Reference'].min()))
    ymax = float(max(frame['Sample'].max(), frame['Reference'].max()))
    pad = max((ymax-ymin)*0.06, 0.1)
    ymin, ymax = ymin-pad, ymax+pad
    low, high = float(frame["Wavenumber"].min()), float(frame["Wavenumber"].max())
    def xp(x):
        return left + (high-x)/(high-low)*(right-left)
    def yp(y):
        return bottom - (y-ymin)/(ymax-ymin)*(bottom-top)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial, sans-serif" font-size="15" fill="#222">']
    for tick in wavenumber_ticks(low, high):
        x=xp(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#e4e8ed"/>')
        parts.append(f'<text x="{x:.2f}" y="{bottom+24}" text-anchor="middle">{tick:g}</text>')
    for tick in np.linspace(ymin,ymax,6):
        y=yp(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#e4e8ed"/>')
        parts.append(f'<text x="{left-10}" y="{y+5:.2f}" text-anchor="end">{tick:.2f}</text>')
    parts.append(f'<path d="M{left} {top} V{bottom} H{right}" fill="none" stroke="#333"/>')
    for i, name in enumerate(['Sample', 'Reference']):
        pts=' '.join(f'{xp(x):.2f},{yp(y):.2f}' for x,y in zip(frame['Wavenumber'],frame[name]))
        dash=' stroke-dasharray="7 4"' if i else ''
        color=SPECTRUM_COLORS[i]
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.7" stroke-linejoin="round"{dash}/>')
        lx=left+i*155
        parts.append(f'<line x1="{lx}" y1="34" x2="{lx+32}" y2="34" stroke="{color}" stroke-width="2"{dash}/>')
        parts.append(f'<text x="{lx+40}" y="39">{name}</text>')
    parts.extend([f'<text x="{(left+right)/2}" y="535" text-anchor="middle">Wavenumber (cm⁻¹)</text>',
                  f'<text transform="translate(25 {(top+bottom)/2}) rotate(-90)" text-anchor="middle">Standardized absorbance</text>', '</g></svg>'])
    return ''.join(parts).encode('utf-8')


def render_spectral_result(result, prefix):
    hits = result['hits']
    best = hits.iloc[0]
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        st.markdown("**Spectral Comparison**")
        frame = spectrum_frame(result)
        low, high = result['range_min_cm1'], result['range_max_cm1']
        long = frame.melt('Wavenumber', var_name='Spectrum', value_name='Standardized absorbance')
        chart = alt.Chart(long).mark_line(strokeWidth=1.7).encode(
            x=alt.X('Wavenumber:Q', title='Wavenumber (cm⁻¹)',
                    scale=alt.Scale(domain=[low, high], reverse=True, nice=False, zero=False),
                    axis=alt.Axis(values=wavenumber_ticks(low, high), format='d', labelOverlap=False)),
            y=alt.Y('Standardized absorbance:Q', scale=alt.Scale(zero=False)),
            color=alt.Color(
                'Spectrum:N',
                scale=alt.Scale(domain=['Sample', 'Reference'], range=SPECTRUM_COLORS),
                legend=alt.Legend(title='Spectrum', orient='right', direction='vertical', symbolStrokeWidth=3)
            ),
            strokeDash=alt.StrokeDash('Spectrum:N', scale=alt.Scale(domain=['Sample', 'Reference'], range=[[1,0],[7,4]]), legend=None),
            tooltip=['Spectrum:N','Wavenumber:Q',alt.Tooltip('Standardized absorbance:Q',format='.3f')]
        ).properties(height=350)
        st.altair_chart(chart,use_container_width=True)
        st.caption('Curves are standardized for visualization; the score is Pearson correlation, not a probability.')
        st.download_button('Download Spectral Comparison (SVG)', spectral_svg(frame),
                           file_name='FTIR_spectral_comparison.svg', mime='image/svg+xml', key=prefix+'_svg')
    with right:
        st.markdown('**Best Reference Match**')
        st.metric('Candidate polymer',str(best['material_group']))
        st.write(polymer_name(best['material_group']))
        st.metric('Spectral similarity (r)',f"{best['similarity']:.4f}")
        st.write(f"Reference: {best['sample_id']}")
        st.caption(f"Source: {best['source']} | ID: {best['library_id']}")
        st.caption(f"Compared with {result['count']:,} usable reference spectra.")
        st.caption(f"Mode: {result['mode']}")
        st.caption(f"Matching range: {high:g}–{low:g} cm⁻¹ | Coverage: {result['coverage_percent']:.1f}%")
        if not result['fingerprint_complete']:
            st.caption('The 650–1500 cm⁻¹ fingerprint region is not fully covered; interpret matches with care.')
    st.markdown('**Top-5 Polymer Matches**')
    table=hits.drop(columns=['material_subtype']).rename(columns={'rank':'Rank','material_group':'Polymer','source':'Source','sample_id':'Reference','library_id':'Library ID','similarity':'Similarity'})
    st.dataframe(table.style.format({'Similarity':'{:.4f}'}),hide_index=True,use_container_width=True)
    st.download_button('Download Spectral Matches CSV',batch_csv_bytes(hits.drop(columns=['material_subtype']).assign(**match_metadata(result))),file_name='FTIR_matches.csv',mime='text/csv',key=prefix+'_csv')
    st.caption('Preliminary matches against selected polymer references; non-plastic materials are not covered, and no validated acceptance threshold has been established.')


def show_ftir():
    st.markdown('Match measured FTIR spectra against selected public FTIR-PLASTIC (c4) and FLOPP-e references.')
    uploaded, signal, mode = spectral_inputs('ftir')
    key=(upload_identity(uploaded),signal,mode) if uploaded is not None else None
    if st.button('Match Spectrum',type='primary',use_container_width=True,key='run_ftir'):
        st.session_state.pop('ftir_result',None)
        if uploaded is None:
            st.warning('Please upload a spectrum.')
        else:
            try:
                with st.spinner('Matching spectrum...'):
                    result=get_library().match(uploaded.getvalue(),uploaded.name,signal,mode=mode)
                st.session_state['ftir_result']=(key,result)
            except Exception as exc:
                st.error(f'Spectral matching failed: {exc}')
    cached=st.session_state.get('ftir_result')
    if cached and cached[0]==key:
        render_spectral_result(cached[1],'ftir')


def batch_match_spectra(uploaded_files, signal, library, progress=None, mode=FULL_RANGE):
    """Apply the selected mode independently to each spectrum; isolate file errors."""
    if not uploaded_files:
        raise ValueError("Please upload one or more spectra.")
    if len(uploaded_files) > MAX_FTIR_BATCH:
        raise ValueError(f"Please select no more than {MAX_FTIR_BATCH} spectra per batch.")
    if sum(f.size for f in uploaded_files) > MAX_FTIR_BATCH_BYTES:
        raise ValueError("Total spectrum file size must not exceed 50 MiB.")
    rows, all_hits, details = [], [], {}
    for i, uploaded in enumerate(uploaded_files, start=1):
        name = Path(uploaded.name).name
        row = dict(sample_no=i, spectrum_name=name, candidate_polymer="",
                   spectral_similarity=None, reference="", reference_source="",
                   library_id="", signal_type="", mode=mode, range_min_cm1=None,
                   range_max_cm1=None, coverage_percent=None, fingerprint_complete=None, error="")
        try:
            result = library.match(uploaded.getvalue(), uploaded.name, signal, mode=mode)
            best = result['hits'].iloc[0]
            row.update(candidate_polymer=str(best['material_group']),
                       spectral_similarity=float(best['similarity']),
                       reference=best['sample_id'], reference_source=best['source'],
                       library_id=best['library_id'], signal_type=result['unit'])
            row.update(match_metadata(result))
            hits = result['hits'].copy().assign(**match_metadata(result))
            hits.insert(0, 'spectrum_name', name)
            hits.insert(0, 'sample_no', i)
            all_hits.append(hits)
            details[i] = result
        except Exception as exc:
            row['error'] = str(exc) or type(exc).__name__
        rows.append(row)
        if progress is not None:
            progress(i / len(uploaded_files), f"Processed {i}/{len(uploaded_files)} spectra")
    hits_columns = ['sample_no', 'spectrum_name', 'rank', 'material_group',
                    'material_subtype', 'source', 'sample_id', 'library_id', 'similarity',
                    'mode', 'range_min_cm1', 'range_max_cm1', 'coverage_percent', 'fingerprint_complete']
    frame = pd.DataFrame(rows)
    for column in ('spectral_similarity', 'range_min_cm1', 'range_max_cm1', 'coverage_percent'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    return (frame,
            pd.concat(all_hits, ignore_index=True) if all_hits else pd.DataFrame(columns=hits_columns),
            details)


def batch_csv_bytes(frame):
    """Avoid interpreting uploaded filenames or other text as spreadsheet formulas."""
    safe = frame.copy()
    for column in safe.select_dtypes(include=['object', 'string']).columns:
        safe[column] = safe[column].map(lambda v: "'" + v if isinstance(v, str) and v.lstrip(' ').startswith(('=', '+', '-', '@', '\t', '\r', '\n')) else v)
    return safe.to_csv(index=False, float_format='%.6f').encode('utf-8-sig')


def show_ftir_batch():
    st.markdown('Match multiple FTIR spectra against the same public reference library.')
    uploaded_files = st.file_uploader(
        'Upload FTIR Spectra (SP, CSV, TXT)', type=['sp', 'csv', 'txt'],
        accept_multiple_files=True, key='ftir_batch_files')
    signal = 'Absorbance'
    if any(Path(f.name).suffix.lower() != '.sp' for f in uploaded_files):
        signal = st.selectbox('Signal type for all CSV/TXT files',
                              ['Absorbance', 'Transmittance (%)', 'Transmittance (fraction)'],
                              key='ftir_batch_unit')
        st.caption('Use separate batches if CSV/TXT files have different signal types.')
    st.caption('Up to 100 files per batch; 5 MiB per file and 50 MiB in total. SP signal units are read automatically. CSV/TXT: two numeric columns, wavenumber in cm⁻¹ and signal.')
    mode = matching_mode('ftir_batch')
    valid = bool(uploaded_files) and len(uploaded_files) <= MAX_FTIR_BATCH and sum(f.size for f in uploaded_files) <= MAX_FTIR_BATCH_BYTES
    key = (tuple(upload_identity(f) for f in uploaded_files), signal, mode) if valid else None
    cached = st.session_state.get('ftir_batch_result')
    if cached is not None and cached[0] != key:
        st.session_state.pop('ftir_batch_result', None)
    if st.button('Match Spectra', type='primary', use_container_width=True, key='run_ftir_batch'):
        st.session_state.pop('ftir_batch_result', None)
        if not valid:
            st.error('Please upload 1–100 spectra with a total size of at most 50 MiB.')
        else:
            bar = st.progress(0, text='Matching spectra...')
            try:
                library = get_library()
                frame, hits, details = batch_match_spectra(uploaded_files, signal, library,
                    progress=lambda value, text: bar.progress(value, text=text), mode=mode)
                st.session_state['ftir_batch_result'] = (key, frame, hits, details)
            except Exception as exc:
                st.error(f'Batch matching failed: {exc}')
            finally:
                bar.empty()
    cached = st.session_state.get('ftir_batch_result')
    if cached is None or cached[0] != key:
        return
    _, frame, hits, details = cached
    success = frame['error'].eq('')
    st.subheader('Batch Matching Results')
    a, b, c = st.columns(3)
    a.metric('Total Spectra', len(frame))
    b.metric('Successful', int(success.sum()))
    c.metric('Failed', int((~success).sum()))
    table = frame.rename(columns={'sample_no':'No.', 'spectrum_name':'Spectrum',
        'candidate_polymer':'Candidate Polymer', 'spectral_similarity':'Similarity',
        'reference':'Reference', 'reference_source':'Source', 'library_id':'Library ID',
        'signal_type':'Signal Type', 'mode':'Mode', 'range_min_cm1':'Min (cm⁻¹)',
        'range_max_cm1':'Max (cm⁻¹)', 'coverage_percent':'Coverage (%)',
        'fingerprint_complete':'Full Fingerprint Region', 'error':'Error'})
    st.dataframe(table.style.format({'Similarity':'{:.4f}', 'Coverage (%)':'{:.1f}', 'Min (cm⁻¹)':'{:g}', 'Max (cm⁻¹)':'{:g}'}, na_rep=''), hide_index=True, use_container_width=True)
    a,b = st.columns(2)
    a.download_button('Download Batch Results CSV', batch_csv_bytes(frame),
        file_name='FTIR_batch_results.csv', mime='text/csv', key='ftir_batch_summary_csv', use_container_width=True)
    b.download_button('Download All Top-5 Matches CSV', batch_csv_bytes(hits.drop(columns=['material_subtype'])),
        file_name='FTIR_batch_top5.csv', mime='text/csv', key='ftir_batch_top5_csv',
        disabled=hits.empty, use_container_width=True)
    st.caption('Successful means a spectrum was processed, not that its material identity was confirmed. Candidate matches are preliminary; similarity is Pearson correlation, not probability.')
    if mode == ADAPTIVE_RANGE:
        st.caption('Similarity scores from different matching ranges are not directly comparable.')
    if not details:
        return
    st.markdown('**Candidate Polymer Counts**')
    counts = frame.loc[success, 'candidate_polymer'].value_counts().rename_axis('Polymer').reset_index(name='Spectra')
    count_chart = alt.Chart(counts).mark_bar(color=SPECTRUM_COLORS[0]).encode(
        x=alt.X('Spectra:Q', axis=alt.Axis(tickMinStep=1)),
        y=alt.Y('Polymer:N', sort='-x'), tooltip=['Polymer:N', 'Spectra:Q'])
    st.altair_chart(count_chart, use_container_width=True)
    st.markdown('**Individual Spectrum Details**')
    names = dict(zip(frame['sample_no'], frame['spectrum_name']))
    selected = st.selectbox('Select a matched spectrum', options=list(details),
        format_func=lambda n: f"{n}. {names[n]}", key='ftir_batch_detail')
    render_spectral_result(details[selected], f'ftir_batch_detail_{selected}')


def show_integrated():
    st.info('Upload a preselected microscopy image and an FTIR spectrum measured from the same particle. The two analyses are independent; this is not a multimodal model.')
    a,b=st.columns(2,gap='large')
    with a:
        image=st.file_uploader('Microscopy Image',type=['jpg','jpeg','png','tif','tiff','bmp','webp'],key='integrated_img')
    with b:
        spectrum,signal,mode=spectral_inputs('integrated')
    key=(upload_identity(image),upload_identity(spectrum),signal,mode) if image is not None and spectrum is not None else None
    if st.button('Analyze Image and Spectrum',type='primary',use_container_width=True,key='run_integrated'):
        st.session_state.pop('integrated_result',None)
        if key is None:
            st.warning('Please upload both files.')
        else:
            try:
                with st.spinner('Analyzing image and spectrum...'):
                    img=open_uploaded(image)
                    predictor=prepare_model()
                    if predictor is not None:
                        pred=predictor.predict(img)
                        result=get_library().match(spectrum.getvalue(),spectrum.name,signal,mode=mode)
                        st.session_state['integrated_result']=(key,pred,result)
            except Exception as exc:
                st.error(f'Analysis failed: {exc}')
    cached=st.session_state.get('integrated_result')
    if cached and cached[0]==key:
        _,pred,result=cached
        best=result['hits'].iloc[0]
        c1,c2=st.columns(2)
        c1.metric('Predicted morphology',pred.label.capitalize())
        c1.metric('Morphology confidence',f'{pred.confidence*100:.2f}%')
        c2.metric('Candidate polymer',str(best['material_group']))
        c2.metric('Spectral similarity (r)',f"{best['similarity']:.4f}")
        with st.expander('Morphology class probabilities'):
            show_probabilities(pred)
        render_spectral_result(result,'integrated')
        row={'image_name':Path(image.name).name,'spectrum_name':Path(spectrum.name).name,'morphology':pred.label,'morphology_confidence':pred.confidence,'candidate_polymer':best['material_group'],'spectral_similarity':best['similarity'],'reference_id':best['library_id'],'reference_source':best['source']}
        st.download_button('Download Integrated Result CSV',batch_csv_bytes(pd.DataFrame([{**row, **match_metadata(result)}])),file_name='integrated_result.csv',mime='text/csv',key='integrated_all_csv')


st.title("CACTNet | Microplastic Identification System")
st.markdown("Morphology classification and FTIR spectral matching for preselected microplastic samples.")
tab_single, tab_batch, tab_ftir, tab_ftir_batch, tab_integrated = st.tabs(["Single Image", "Batch Classification", "FTIR Spectral Matching", "Batch FTIR Matching", "Integrated Identification"])
with tab_single:
    show_single()
with tab_batch:
    show_batch()
with tab_ftir:
    show_ftir()
with tab_ftir_batch:
    show_ftir_batch()
with tab_integrated:
    show_integrated()
