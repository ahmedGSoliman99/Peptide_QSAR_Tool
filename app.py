"""Streamlit application for peptide-focused QSAR modeling and prediction."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.descriptor_engine import (
    DescriptorConfig,
    calculate_descriptors_dataframe,
    describe_descriptor_set,
    generate_peptide_3d_mol,
    get_descriptor_columns,
)
from modules.design_suggestions import (
    peptide_activity_design_profile,
    sequence_alignment_to_reference,
    infer_optimization_direction,
    positional_activity_analysis,
    suggest_mutations_for_sequence,
)
from modules.io_utils import (
    guess_sequence_column,
    guess_target_columns,
    merge_manual_and_uploaded,
    parse_sequence_block,
    read_uploaded_table,
    validate_sequences,
)
from modules.modeling import (
    available_models_for_task,
    compute_model_feature_importance,
    compute_shap_importance,
    get_model_family,
    load_model_bundle,
    predict_with_bundle,
    save_model_bundle,
    summarize_model_bundle,
    train_and_compare_models,
)
from modules.preprocessing import PreprocessingConfig, SplitConfig, compute_embedding, compute_pca_analysis
from modules.reporting import dataframe_to_csv_bytes, dataframe_to_excel_bytes, generate_html_report
from modules.structure_criteria import evaluate_structure_criteria, summarize_structure_quality
from modules.visualization import (
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_descriptor_distribution,
    plot_embedding,
    plot_feature_importance,
    plot_model_comparison,
    plot_pca_explained_variance,
    plot_pca_loading_bar,
    plot_position_recommendation_scores,
    plot_position_residue_heatmap,
    plot_prediction_ranking,
    plot_residuals,
    plot_structure_criteria_radar,
    plot_structure_quality_distribution,
)


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "saved_models"
DOCS_DIR = ROOT_DIR / "docs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEVELOPER_NAME = "Ahmed G. Soliman"
DEVELOPER_PORTFOLIO = "https://sites.google.com/view/ahmed-g-soliman/home"
DEVELOPER_PROFILE = {
    "Developer": DEVELOPER_NAME,
    "Current role": "MEXT master's student at Kyutech, Japan, School of Life Science and Engineering",
    "Background": "Biotechnology Program, Faculty of Agriculture, Ain Shams University, Cairo, Egypt",
    "Experience": "Previous instructor at ACGEB in in-silico drug design and immune-informatics",
    "Scopus ID": "58569160700",
    "ResearcherID (WOS)": "ABE-8406-2021",
    "ORCID": "0000-0002-1122-3993",
    "Portfolio": DEVELOPER_PORTFOLIO,
}


st.set_page_config(
    page_title="Peptide QSAR Prediction Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: "Manrope", "Segoe UI", sans-serif;
}

.main > div {
    padding-top: 1rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #edf7ff 0%, #f7fcff 100%);
    border-right: 1px solid #d8e7f5;
}

.hero {
    border: 1px solid #d4e6f7;
    background: linear-gradient(135deg, #f5fbff 0%, #e8f4ff 60%, #dff0ff 100%);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

.hero h1 {
    margin: 0;
    color: #0b3a5e;
    font-size: 1.8rem;
}

.hero p {
    margin: 0.4rem 0 0 0;
    color: #234867;
    font-size: 0.98rem;
}

.soft-card {
    border: 1px solid #d7e7f5;
    border-radius: 14px;
    padding: 0.9rem;
    background: #ffffff;
    box-shadow: 0 2px 10px rgba(20, 68, 107, 0.06);
}

.small-note {
    color: #335874;
    font-size: 0.88rem;
}
</style>
"""
st.markdown(CUSTOM_STYLE, unsafe_allow_html=True)


TASK_DISPLAY_TO_VALUE = {
    "Regression": "regression",
    "Binary Classification": "binary_classification",
    "Multiclass Classification": "multiclass_classification",
}
TASK_VALUE_TO_DISPLAY = {v: k for k, v in TASK_DISPLAY_TO_VALUE.items()}


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "uploaded_df": None,
        "uploaded_filename": None,
        "manual_input_text": "",
        "validated_df": None,
        "invalid_df": None,
        "duplicate_df": None,
        "task_type": "regression",
        "target_column": None,
        "descriptor_df": None,
        "descriptor_columns": [],
        "descriptor_config": DescriptorConfig(),
        "training_result": None,
        "active_model_name": None,
        "loaded_bundle": None,
        "prediction_df": None,
        "prediction_descriptors_df": None,
        "last_report_html": None,
        "embedding_df": None,
        "pca_analysis": None,
        "prediction_quality_df": None,
        "design_analysis": None,
        "peptide_alignment_analysis": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _sanitize_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name.strip())
    return name or "model_bundle.joblib"


def _load_example_df() -> pd.DataFrame:
    path = DATA_DIR / "example_peptides.csv"
    if not path.exists():
        raise FileNotFoundError("Example dataset is missing.")
    return pd.read_csv(path)


def _model_file_list() -> list[Path]:
    return sorted(MODEL_DIR.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)


def _clear_downstream_after_new_input() -> None:
    st.session_state["descriptor_df"] = None
    st.session_state["descriptor_columns"] = []
    st.session_state["training_result"] = None
    st.session_state["active_model_name"] = None
    st.session_state["prediction_df"] = None
    st.session_state["prediction_descriptors_df"] = None
    st.session_state["last_report_html"] = None
    st.session_state["embedding_df"] = None
    st.session_state["pca_analysis"] = None
    st.session_state["prediction_quality_df"] = None
    st.session_state["design_analysis"] = None
    st.session_state["peptide_alignment_analysis"] = None


def _plot(fig, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, key=key)


def _plot_peptide_3d(sequence: str):
    mol3d, status = generate_peptide_3d_mol(sequence)
    if mol3d is None:
        return px.scatter_3d(title=f"3D peptide view unavailable: {status}", template="plotly_white"), status
    conf = mol3d.GetConformer()
    atoms = []
    for atom in mol3d.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(
            {
                "Atom": atom.GetSymbol(),
                "Index": atom.GetIdx(),
                "x": pos.x,
                "y": pos.y,
                "z": pos.z,
                "AtomicNum": atom.GetAtomicNum(),
            }
        )
    atom_df = pd.DataFrame(atoms)
    fig = px.scatter_3d(
        atom_df,
        x="x",
        y="y",
        z="z",
        color="Atom",
        hover_name="Atom",
        hover_data={"Index": True, "x": ":.2f", "y": ":.2f", "z": ":.2f"},
        template="plotly_white",
        title="Approximate peptide 3D conformer (ETKDG/MMFF or UFF)",
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color="#102a43")))
    for bond in mol3d.GetBonds():
        begin = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        end = conf.GetAtomPosition(bond.GetEndAtomIdx())
        fig.add_trace(
            {
                "type": "scatter3d",
                "x": [begin.x, end.x],
                "y": [begin.y, end.y],
                "z": [begin.z, end.z],
                "mode": "lines",
                "line": {"color": "#7f8c8d", "width": 3},
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )
    fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=45, b=0))
    return fig, status


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Peptide QSAR Prediction Tool</h1>
          <p>Beginner-friendly platform for peptide descriptor engineering, QSAR model training,
          explainable prediction, and scientific report export.</p>
          <p><b>Developed by Ahmed G. Soliman.</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_home_tab() -> None:
    _render_header()
    col1, col2 = st.columns([1.35, 1.0], gap="large")
    with col1:
        st.markdown("### Workflow")
        st.markdown(
            """
1. Upload peptide sequences (manual, FASTA, CSV/Excel/TXT/SDF)
2. Validate sequences and remove duplicates/invalid residues
3. Compute peptide descriptors (physicochemical + composition)
4. Train and compare QSAR/ML models
5. Evaluate model performance with scientific metrics
6. Predict new peptides and rank by activity/probability
7. Visualize descriptors, embeddings, feature influence, and SHAP
8. Export tables, trained models, and HTML report
            """
        )
        st.markdown("### Tips for beginners")
        st.markdown(
            """
- Start by clicking **Load Example Dataset** in the Upload tab.
- For regression targets, use numeric columns like `Activity`, `IC50`, or `DockingScore`.
- For classification, use categorical targets like `Class` or `Toxicity`.
- Save your best model so you can reuse it later without retraining.
            """
        )
    with col2:
        st.markdown("### Included Descriptor Families")
        st.markdown(
            """
- Amino acid composition (AAC)
- Optional dipeptide composition
- Sequence length and Shannon entropy
- Molecular weight, aromaticity, pI, net charge
- Instability index, GRAVY/hydrophobicity
- Aliphatic and Boman indices
- Hydrophobic moment, helix/sheet propensity, charge density
- Residue class fractions (positive/negative/polar/etc.)
- Optional elemental composition and motif fingerprints
            """
        )
        st.markdown(
            "<div class='soft-card'><b>Windows Launch:</b> Use <code>run_app.bat</code> for one-click startup.</div>",
            unsafe_allow_html=True,
        )


def _render_upload_tab() -> None:
    st.subheader("1) Upload Data")
    st.caption("Input one sequence, many sequences, or files with optional target properties.")

    left, right = st.columns([1.1, 1.0], gap="large")
    with left:
        st.markdown("#### Manual Sequence Input")
        manual_text = st.text_area(
            "Paste peptide sequences (plain list, comma-separated, or FASTA):",
            value=st.session_state["manual_input_text"],
            height=170,
            help="Example: AKLVFF, GIGAVLKVLTTGLPALISWIKRKRQQ, or FASTA entries.",
        )
        st.session_state["manual_input_text"] = manual_text
        manual_df = parse_sequence_block(manual_text)
        st.markdown(
            f"<div class='small-note'>Detected manual sequences: <b>{len(manual_df)}</b></div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### File Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV / Excel / TXT / FASTA / SDF",
            type=["csv", "xlsx", "xls", "txt", "fasta", "fa", "faa", "sdf"],
            key="upload_dataset_file",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load Example Dataset", use_container_width=True):
                try:
                    ex_df = _load_example_df()
                    st.session_state["uploaded_df"] = ex_df
                    st.session_state["uploaded_filename"] = "example_peptides.csv"
                    st.success("Example dataset loaded.")
                except Exception as exc:
                    st.error(f"Could not load example dataset: {exc}")
        with c2:
            if st.button("Clear Uploaded File", use_container_width=True):
                st.session_state["uploaded_df"] = None
                st.session_state["uploaded_filename"] = None

        if uploaded_file is not None:
            try:
                file_df = read_uploaded_table(uploaded_file.name, uploaded_file.getvalue())
                st.session_state["uploaded_df"] = file_df
                st.session_state["uploaded_filename"] = uploaded_file.name
                st.success(f"Loaded `{uploaded_file.name}` with {len(file_df)} rows.")
            except Exception as exc:
                st.error(f"Could not read file: {exc}")

        if isinstance(st.session_state["uploaded_df"], pd.DataFrame):
            st.dataframe(st.session_state["uploaded_df"].head(15), use_container_width=True, height=260)
            st.caption(
                f"Active dataset: {st.session_state.get('uploaded_filename') or 'uploaded table'} "
                f"({len(st.session_state['uploaded_df'])} rows)"
            )

    uploaded_df = st.session_state["uploaded_df"]
    uploaded_sequence_col = None
    target_default = None

    if isinstance(uploaded_df, pd.DataFrame) and not uploaded_df.empty:
        guessed_seq_col = guess_sequence_column(uploaded_df)
        sequence_options = list(uploaded_df.columns)
        default_idx = sequence_options.index(guessed_seq_col) if guessed_seq_col in sequence_options else 0
        uploaded_sequence_col = st.selectbox(
            "Select sequence column in uploaded table:",
            options=sequence_options,
            index=default_idx,
        )
        target_hints = guess_target_columns(uploaded_df)
        if target_hints:
            target_default = target_hints[0]

    task_labels = list(TASK_DISPLAY_TO_VALUE.keys())
    default_task_idx = task_labels.index(TASK_VALUE_TO_DISPLAY.get(st.session_state["task_type"], "Regression"))
    selected_task_label = st.selectbox("Task type:", task_labels, index=default_task_idx)
    selected_task_value = TASK_DISPLAY_TO_VALUE[selected_task_label]

    merged_preview = merge_manual_and_uploaded(uploaded_df, manual_df, sequence_col_uploaded=uploaded_sequence_col)
    target_options = [
        col for col in merged_preview.columns if col.lower() not in {"sequence", "name"}
    ]
    if target_options:
        initial_target = (
            st.session_state["target_column"]
            if st.session_state["target_column"] in target_options
            else (target_default if target_default in target_options else target_options[0])
        )
        selected_target = st.selectbox("Target/property column (for model training):", target_options, index=target_options.index(initial_target))
        st.caption("Tip: docking outputs such as DockingScore / Vina / Affinity can be used as regression targets.")
    else:
        selected_target = None
        st.info("No target column detected yet. You can still compute descriptors and run prediction-only workflows.")

    if st.button("Validate And Save Input Data", type="primary"):
        if merged_preview.empty:
            st.warning("Please provide at least one peptide sequence (manual input or file).")
            return
        if "Sequence" not in merged_preview.columns:
            st.error("Could not detect a sequence column. Please choose the correct sequence field.")
            return

        validation = validate_sequences(merged_preview, sequence_col="Sequence")
        st.session_state["validated_df"] = validation.valid_df
        st.session_state["invalid_df"] = validation.invalid_df
        st.session_state["duplicate_df"] = validation.duplicate_df
        st.session_state["task_type"] = selected_task_value
        st.session_state["target_column"] = selected_target
        _clear_downstream_after_new_input()

        st.success(f"Saved {len(validation.valid_df)} valid unique peptide sequences.")
        if not validation.invalid_df.empty:
            st.warning(f"Invalid sequences found: {len(validation.invalid_df)}")
        if not validation.duplicate_df.empty:
            st.info(f"Duplicate sequences detected: {len(validation.duplicate_df)}")

    if isinstance(st.session_state["validated_df"], pd.DataFrame):
        st.markdown("#### Validation Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Valid Unique Peptides", int(len(st.session_state["validated_df"])))
        c2.metric("Invalid Sequences", int(len(st.session_state["invalid_df"])) if st.session_state["invalid_df"] is not None else 0)
        c3.metric("Duplicates Found", int(len(st.session_state["duplicate_df"])) if st.session_state["duplicate_df"] is not None else 0)
        st.dataframe(st.session_state["validated_df"].head(20), use_container_width=True, height=260)

        if isinstance(st.session_state["invalid_df"], pd.DataFrame) and not st.session_state["invalid_df"].empty:
            with st.expander("Show invalid sequences"):
                st.dataframe(st.session_state["invalid_df"], use_container_width=True)

        if isinstance(st.session_state["duplicate_df"], pd.DataFrame) and not st.session_state["duplicate_df"].empty:
            with st.expander("Show duplicate sequences"):
                st.dataframe(st.session_state["duplicate_df"], use_container_width=True)


def _render_descriptors_tab() -> None:
    st.subheader("2) Descriptor Calculation")
    validated_df = st.session_state["validated_df"]
    if not isinstance(validated_df, pd.DataFrame) or validated_df.empty:
        st.info("Validate input data first in the Upload tab.")
        return

    c1, c2, c3, c4 = st.columns(4)
    # User-customizable descriptor toggles:
    # add new switches here if you later introduce more descriptor families.
    include_dipeptide = c1.checkbox("Include dipeptide composition (400 features)", value=False)
    include_fingerprints = c2.checkbox("Include motif fingerprints", value=True)
    include_elemental = c3.checkbox("Include elemental composition", value=True)
    include_3d = c4.checkbox(
        "Include approximate 3D descriptors",
        value=False,
        help="Uses RDKit MolFromFASTA + ETKDG to generate a 3D peptide conformer and shape descriptors. Slower; skipped for peptides longer than 45 residues.",
    )
    config = DescriptorConfig(
        include_dipeptide=include_dipeptide,
        include_fingerprints=include_fingerprints,
        include_elemental=include_elemental,
        include_3d=include_3d,
    )
    st.caption(f"Descriptor set: {describe_descriptor_set(config)}")

    if st.button("Calculate Descriptors", type="primary"):
        try:
            progress = st.progress(0)
            status = st.empty()

            def _callback(i: int, total: int) -> None:
                pct = int((i / max(total, 1)) * 100)
                progress.progress(min(max(pct, 0), 100))
                status.caption(f"Calculating descriptors: {i}/{total}")

            descriptor_df = calculate_descriptors_dataframe(
                validated_df,
                sequence_col="Sequence",
                config=config,
                progress_callback=_callback,
            )
            descriptor_columns = get_descriptor_columns(descriptor_df)
            st.session_state["descriptor_df"] = descriptor_df
            st.session_state["descriptor_columns"] = descriptor_columns
            st.session_state["descriptor_config"] = config
            st.session_state["training_result"] = None
            st.session_state["active_model_name"] = None
            st.session_state["prediction_df"] = None
            st.session_state["prediction_descriptors_df"] = None
            st.success(f"Calculated descriptors for {len(descriptor_df)} peptides.")
        except Exception as exc:
            st.error(f"Descriptor calculation failed: {exc}")

    descriptor_df = st.session_state["descriptor_df"]
    descriptor_cols = st.session_state["descriptor_columns"]
    if isinstance(descriptor_df, pd.DataFrame) and not descriptor_df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Peptides With Descriptors", int(len(descriptor_df)))
        c2.metric("Descriptor Features", int(len(descriptor_cols)))
        c3.metric("Total Columns", int(descriptor_df.shape[1]))
        st.dataframe(descriptor_df.head(20), use_container_width=True, height=320)

        csv_bytes = dataframe_to_csv_bytes(descriptor_df)
        excel_bytes = dataframe_to_excel_bytes({"Descriptors": descriptor_df})
        d1, d2 = st.columns(2)
        d1.download_button(
            "Download Descriptor Table (CSV)",
            data=csv_bytes,
            file_name="peptide_descriptors.csv",
            mime="text/csv",
            use_container_width=True,
        )
        d2.download_button(
            "Download Descriptor Table (Excel)",
            data=excel_bytes,
            file_name="peptide_descriptors.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


def _render_train_tab() -> None:
    st.subheader("3) Train Model")
    descriptor_df = st.session_state["descriptor_df"]
    descriptor_cols = st.session_state["descriptor_columns"]

    if not isinstance(descriptor_df, pd.DataFrame) or descriptor_df.empty:
        st.info("Calculate descriptors first in the Descriptors tab.")
        return

    target_candidates = [
        col
        for col in descriptor_df.columns
        if col not in descriptor_cols and col.lower() not in {"sequence", "name"}
    ]
    # User-customizable target selection:
    # choose which property column drives QSAR training (Activity, IC50, Toxicity, etc.).
    if not target_candidates:
        st.warning("No target/property column found. Upload a dataset with a target (Activity, IC50, Class, etc.).")
        return

    default_target = (
        st.session_state["target_column"]
        if st.session_state["target_column"] in target_candidates
        else target_candidates[0]
    )
    target_col = st.selectbox(
        "Target column for training:",
        target_candidates,
        index=target_candidates.index(default_target),
    )
    st.session_state["target_column"] = target_col

    task_labels = list(TASK_DISPLAY_TO_VALUE.keys())
    default_task_label = TASK_VALUE_TO_DISPLAY.get(st.session_state["task_type"], "Regression")
    task_label = st.selectbox(
        "Task type",
        task_labels,
        index=task_labels.index(default_task_label),
    )
    task_type = TASK_DISPLAY_TO_VALUE[task_label]
    st.session_state["task_type"] = task_type

    y_non_null = descriptor_df[target_col].dropna()
    n_classes = int(y_non_null.astype(str).nunique()) if task_type != "regression" else 1
    model_catalog = available_models_for_task(task_type, n_classes=max(2, n_classes))
    model_names = list(model_catalog.keys())
    if task_type == "regression":
        preferred = ["Gradient Boosting", "Extra Trees", "Random Forest", "Ridge", "SVR (RBF)"]
    else:
        preferred = ["Extra Trees", "Random Forest", "SVM (RBF)", "Gradient Boosting", "Logistic Regression"]
    default_models = [m for m in preferred if m in model_names]
    if not default_models:
        default_models = model_names[: min(5, len(model_names))]
    selected_models = st.multiselect(
        "Models to train and compare:",
        options=model_names,
        default=default_models,
    )
    model_family_df = pd.DataFrame(
        {"Model": model_names, "Family": [get_model_family(m) for m in model_names]}
    )
    with st.expander("Show available model families"):
        st.dataframe(model_family_df, use_container_width=True, hide_index=True)

    with st.expander("Preprocessing & Validation Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        impute_strategy = col1.selectbox("Missing value strategy", ["median", "mean", "most_frequent"], index=0)
        scaler = col2.selectbox("Feature scaling", ["standard", "minmax", "robust", "none"], index=0)
        use_pca = col3.checkbox("Use PCA before modeling", value=False)

        col4, col5, col6 = st.columns(3)
        variance_threshold = col4.number_input("Variance threshold", min_value=0.0, max_value=10.0, value=0.0, step=0.01)
        correlation_threshold = col5.slider("Correlation filter threshold", min_value=0.70, max_value=0.999, value=0.95, step=0.01)
        pca_components = col6.number_input("PCA components", min_value=2, max_value=300, value=20, step=1)

        col7, col8, col9, col10 = st.columns(4)
        test_size = col7.slider("Test size", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
        val_size = col8.slider("Validation size", min_value=0.0, max_value=0.30, value=0.10, step=0.05)
        cv_folds = col9.slider("CV folds", min_value=2, max_value=10, value=5, step=1)
        random_state = col10.number_input("Random seed", min_value=1, max_value=99999, value=42, step=1)
        auto_stabilize = st.checkbox(
            "Auto-stabilize small or high-dimensional QSAR datasets",
            value=True,
            help=(
                "Recommended for peptide QSAR. If descriptors greatly outnumber samples, "
                "the app tightens correlation filtering and uses a compact PCA space to reduce overfitting."
            ),
        )

    if st.button("Train & Compare Models", type="primary"):
        if not selected_models:
            st.warning("Select at least one model.")
            return
        try:
            with st.spinner("Training models and running evaluation..."):
                descriptor_count = len(st.session_state.get("descriptor_columns", []))
                valid_target_rows = int(descriptor_df[target_col].notna().sum())
                effective_use_pca = bool(use_pca)
                effective_pca_components = int(pca_components)
                effective_correlation_threshold = float(correlation_threshold)
                auto_note = ""
                if (
                    auto_stabilize
                    and task_type == "regression"
                    and valid_target_rows > 0
                    and descriptor_count > max(30, valid_target_rows * 2)
                ):
                    effective_use_pca = True
                    effective_pca_components = max(2, min(25, descriptor_count, max(2, valid_target_rows // 4)))
                    effective_correlation_threshold = min(effective_correlation_threshold, 0.90)
                    auto_note = (
                        f"Auto-stabilizer enabled: {descriptor_count} descriptors for {valid_target_rows} rows. "
                        f"Using PCA={effective_pca_components} and correlation threshold={effective_correlation_threshold:.2f}."
                    )
                preprocessing_config = PreprocessingConfig(
                    impute_strategy=impute_strategy,
                    scaler=scaler,
                    variance_threshold=float(variance_threshold),
                    correlation_threshold=effective_correlation_threshold,
                    use_pca=effective_use_pca,
                    pca_components=effective_pca_components,
                )
                split_config = SplitConfig(
                    test_size=float(test_size),
                    val_size=float(val_size),
                    random_state=int(random_state),
                )
                training_result = train_and_compare_models(
                    descriptor_df=descriptor_df,
                    target_column=target_col,
                    task_type=task_type,
                    selected_models=selected_models,
                    descriptor_config=st.session_state["descriptor_config"],
                    preprocessing_config=preprocessing_config,
                    split_config=split_config,
                    cv_folds=int(cv_folds),
                )
                if auto_note:
                    training_result["auto_stabilizer_note"] = auto_note
            st.session_state["training_result"] = training_result
            st.session_state["active_model_name"] = training_result["best_model_name"]
            st.success(f"Training completed. Best model: {training_result['best_model_name']}")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    result = st.session_state["training_result"]
    if isinstance(result, dict):
        comparison_df = result["comparison_table"]
        if result.get("auto_stabilizer_note"):
            st.info(result["auto_stabilizer_note"])
        st.dataframe(comparison_df, use_container_width=True)

        metric_col = (
            "FitQuality_0_100"
            if result["task_type"] == "regression" and "FitQuality_0_100" in comparison_df.columns
            else ("R2" if result["task_type"] == "regression" else "F1")
        )
        fig = plot_model_comparison(comparison_df, primary_metric=metric_col)
        _plot(fig, key="train_model_comparison")
        if result["task_type"] == "regression" and metric_col == "FitQuality_0_100":
            st.caption(
                "The leaderboard uses repeated cross-validation when possible. "
                "FitQuality_0_100 is non-negative for quick comparison; R2/RMSE/MAE remain available for scientific reporting."
            )
        if result["task_type"] == "regression" and "R2" in comparison_df.columns:
            best_r2 = pd.to_numeric(comparison_df["R2"], errors="coerce").max()
            best_cv_r2 = (
                pd.to_numeric(comparison_df["CV_r2_mean"], errors="coerce").max()
                if "CV_r2_mean" in comparison_df.columns
                else np.nan
            )
            if pd.notna(best_r2) and best_r2 < 0:
                if pd.notna(best_cv_r2) and best_cv_r2 >= 0:
                    st.info(
                        "The single holdout test split has negative R2, but repeated cross-validation is non-negative. "
                        "For small peptide QSAR datasets, use CV_R2_mean, RMSE, MAE, and external validation as the main reliability checks."
                    )
                else:
                    st.warning(
                        "Both the holdout test split and cross-validation have weak or negative R2. "
                        "This usually means the current dataset is too small, noisy, or descriptor-heavy for reliable regression. "
                        "Add more measured peptides, reduce descriptors, use PCA/correlation filtering, or validate with an external test set."
                    )
            elif pd.to_numeric(comparison_df["R2"], errors="coerce").lt(0).any():
                st.info(
                    "Some models have negative R2, but the leaderboard picked a better model above them. "
                    "Use the best model row for prediction instead of the weak negative-R2 models."
                )

        st.markdown("#### Save Trained Model")
        model_name = st.selectbox("Choose trained model to save:", list(result["model_outputs"].keys()))
        default_name = _sanitize_filename(
            f"{model_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        )
        model_filename = st.text_input("Model file name:", value=default_name)
        if st.button("Save Model Bundle", use_container_width=True):
            try:
                bundle = result["model_outputs"][model_name]["bundle"]
                save_path = save_model_bundle(bundle, MODEL_DIR / _sanitize_filename(model_filename))
                st.success(f"Model saved: {save_path.name}")
            except Exception as exc:
                st.error(f"Could not save model: {exc}")


def _render_evaluate_tab() -> None:
    st.subheader("4) Evaluate")
    result = st.session_state["training_result"]
    if not isinstance(result, dict):
        st.info("Train at least one model first.")
        return

    model_name = st.selectbox("Select trained model:", list(result["model_outputs"].keys()), key="eval_model_selector")
    model_output = result["model_outputs"][model_name]
    task_type = result["task_type"]
    bundle = model_output["bundle"]

    metrics = model_output["test"]["metrics"]
    st.markdown("#### Test Set Metrics")
    metric_cols = st.columns(5)
    if task_type == "regression":
        metric_cols[0].metric("R2", f"{metrics['R2']:.4f}")
        metric_cols[1].metric("RMSE", f"{metrics['RMSE']:.4f}")
        metric_cols[2].metric("MAE", f"{metrics['MAE']:.4f}")
        if metrics.get("R2", 0) < 0:
            cv_r2 = bundle.cv_summary.get("CV_r2_mean", np.nan)
            if pd.notna(cv_r2) and cv_r2 >= 0:
                st.info(
                    "The holdout R2 is negative for this particular split, but repeated cross-validation is non-negative. "
                    "Use CV_R2_mean together with RMSE/MAE and external validation for the scientific conclusion."
                )
            else:
                st.warning(
                    "R2 is negative, which means this model is performing worse than simply predicting "
                    "the average target value on the test set. Try more training data, fewer descriptors, "
                    "PCA/correlation filtering, or a nonlinear model such as SVR, kNN, Random Forest, or Extra Trees."
                )
    else:
        metric_cols[0].metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        metric_cols[1].metric("Precision", f"{metrics['Precision']:.4f}")
        metric_cols[2].metric("Recall", f"{metrics['Recall']:.4f}")
        metric_cols[3].metric("F1", f"{metrics['F1']:.4f}")
        if not pd.isna(metrics.get("ROC_AUC", np.nan)):
            metric_cols[4].metric("ROC-AUC", f"{metrics['ROC_AUC']:.4f}")

    if task_type == "regression":
        y_true = np.asarray(model_output["test"]["y_true"], dtype=float)
        y_pred = np.asarray(model_output["test"]["y_pred"], dtype=float)
        c1, c2 = st.columns(2)
        with c1:
            _plot(plot_actual_vs_predicted(y_true, y_pred), key="eval_actual_vs_pred")
        with c2:
            _plot(plot_residuals(y_true, y_pred), key="eval_residual_plot")
    else:
        labels = metrics.get("ClassLabels", [])
        cm = np.asarray(metrics.get("ConfusionMatrix"))
        _plot(plot_confusion_matrix(cm, labels), key="eval_confusion_matrix")
        report_dict = metrics.get("ClassificationReport")
        if isinstance(report_dict, dict):
            try:
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df, use_container_width=True, height=260)
            except Exception:
                pass

    cv_cols = [c for c in result["comparison_table"].columns if c.startswith("CV_")]
    if cv_cols:
        st.markdown("#### Cross-Validation Summary")
        st.dataframe(result["comparison_table"][["Model"] + cv_cols], use_container_width=True)

    st.markdown("#### Model Summary")
    summary = summarize_model_bundle(bundle)
    model_family = str(summary.get("ModelFamily", "Unknown"))
    summary_table = pd.DataFrame(
        [
            {"Item": "Model", "Value": summary.get("ModelName", "Unknown")},
            {"Item": "Family", "Value": model_family},
            {"Item": "Task", "Value": summary.get("TaskType", "Unknown")},
            {"Item": "Target", "Value": summary.get("TargetColumn", "Unknown")},
            {"Item": "Descriptors", "Value": summary.get("DescriptorCount", "Unknown")},
            {"Item": "Transformed Features", "Value": summary.get("TransformedFeatureCount", "Unknown")},
            {"Item": "Created At", "Value": summary.get("CreatedAt", "Unknown")},
            {"Item": "Estimator Class", "Value": summary.get("EstimatorClass", "Unknown")},
        ]
    )
    st.dataframe(summary_table, use_container_width=True, hide_index=True)
    with st.expander("Show model hyperparameters"):
        st.json(summary.get("EstimatorParams", {}))

    if model_family in {"Nonlinear Kernel", "Nonlinear Distance"}:
        st.info(
            "This is a nonlinear model. It can capture curved structure-activity relationships "
            "that linear models might miss."
        )


def _resolve_bundle_for_prediction() -> tuple[Any | None, str]:
    result = st.session_state["training_result"]
    use_mode = st.radio(
        "Prediction model source:",
        ["Use model trained in this session", "Load model from saved file"],
        horizontal=True,
    )

    if use_mode == "Use model trained in this session":
        if not isinstance(result, dict):
            return None, "No trained model is available in this session."
        model_names = list(result["model_outputs"].keys())
        default_model = st.session_state.get("active_model_name")
        if default_model not in model_names:
            default_model = model_names[0]
        selected_model = st.selectbox("Select session model:", model_names, index=model_names.index(default_model))
        st.session_state["active_model_name"] = selected_model
        return result["model_outputs"][selected_model]["bundle"], ""

    files = _model_file_list()
    if not files:
        return None, "No saved model files were found in `saved_models/`."
    options = [p.name for p in files]
    chosen = st.selectbox("Choose saved model file:", options)
    if st.button("Load Selected Model"):
        try:
            bundle = load_model_bundle(MODEL_DIR / chosen)
            st.session_state["loaded_bundle"] = bundle
            st.success(f"Loaded model: {chosen}")
        except Exception as exc:
            st.error(f"Could not load model: {exc}")
    if st.session_state.get("loaded_bundle") is None:
        return None, "Load a model file to start prediction."
    return st.session_state["loaded_bundle"], ""


def _render_predict_tab() -> None:
    st.subheader("5) Predict New Peptides")
    bundle, message = _resolve_bundle_for_prediction()
    if bundle is None:
        st.info(message)
        return

    st.caption(
        f"Loaded model: `{bundle.model_name}` | Task: `{TASK_VALUE_TO_DISPLAY.get(bundle.task_type, bundle.task_type)}`"
    )

    left, right = st.columns(2)
    with left:
        pred_text = st.text_area(
            "Input new peptide sequences (line-separated / FASTA / comma-separated):",
            height=170,
            key="prediction_manual_input",
        )
        manual_pred_df = parse_sequence_block(pred_text)
        st.caption(f"Manual prediction entries: {len(manual_pred_df)}")

    with right:
        pred_file = st.file_uploader(
            "Or upload prediction file (CSV/Excel/TXT/FASTA/SDF)",
            type=["csv", "xlsx", "xls", "txt", "fasta", "fa", "faa", "sdf"],
            key="prediction_file",
        )
        uploaded_pred_df = None
        sequence_col = None
        if pred_file is not None:
            try:
                uploaded_pred_df = read_uploaded_table(pred_file.name, pred_file.getvalue())
                seq_guess = guess_sequence_column(uploaded_pred_df)
                cols = list(uploaded_pred_df.columns)
                idx = cols.index(seq_guess) if seq_guess in cols else 0
                sequence_col = st.selectbox("Prediction file sequence column:", cols, index=idx, key="pred_seq_col")
                st.dataframe(uploaded_pred_df.head(12), use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read prediction file: {exc}")

    merged_pred = merge_manual_and_uploaded(uploaded_pred_df, manual_pred_df, sequence_col_uploaded=sequence_col)
    if "Sequence" in merged_pred.columns and not merged_pred.empty:
        validation = validate_sequences(merged_pred, sequence_col="Sequence")
        valid_pred_df = validation.valid_df
        invalid_count = len(validation.invalid_df)
        duplicate_count = len(validation.duplicate_df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Valid peptides", int(len(valid_pred_df)))
        c2.metric("Invalid", int(invalid_count))
        c3.metric("Duplicates", int(duplicate_count))
    else:
        valid_pred_df = pd.DataFrame(columns=["Sequence"])
        if pred_file is not None and uploaded_pred_df is not None:
            st.warning(
                "The uploaded SDF/table did not expose a usable peptide sequence column. "
                "For peptide QSAR descriptors, the file must contain a sequence property such as "
                "`Sequence`, `Peptide Sequence`, `AA Sequence`, or another column containing one-letter amino acid codes."
            )

    if st.button("Run Prediction", type="primary"):
        if valid_pred_df.empty:
            st.warning("No valid sequences to predict.")
            return
        try:
            if "Name" not in valid_pred_df.columns:
                valid_pred_df = valid_pred_df.copy()
                valid_pred_df["Name"] = [f"Peptide_{i+1}" for i in range(len(valid_pred_df))]
            sequence_df = valid_pred_df[["Name", "Sequence"]]
            prediction_df, prediction_descriptor_df = predict_with_bundle(bundle, sequence_df)
            criteria_df = evaluate_structure_criteria(prediction_descriptor_df)

            if isinstance(criteria_df, pd.DataFrame) and not criteria_df.empty:
                pass_cols = [c for c in criteria_df.columns if c.startswith("Pass_")]
                criteria_extra = [
                    "StructureQualityScore",
                    "StructureAssessment",
                    "CriteriaPassCount",
                    "CriteriaTotal",
                ] + pass_cols
                join_cols = [c for c in ["Name", "Sequence"] if c in criteria_df.columns and c in prediction_df.columns]
                available_cols = join_cols + [c for c in criteria_extra if c in criteria_df.columns]
                if available_cols:
                    prediction_df = prediction_df.merge(
                        criteria_df[available_cols],
                        on=join_cols,
                        how="left",
                    )

            # Keep a dedicated criteria table aligned with prediction ranking.
            if isinstance(criteria_df, pd.DataFrame) and not criteria_df.empty:
                join_cols_rank = [c for c in ["Name", "Sequence"] if c in criteria_df.columns and c in prediction_df.columns]
                if join_cols_rank:
                    criteria_rank_df = prediction_df[join_cols_rank + ["Rank"]].merge(
                        criteria_df,
                        on=join_cols_rank,
                        how="left",
                    )
                else:
                    criteria_rank_df = criteria_df.copy()
            else:
                criteria_rank_df = pd.DataFrame()
            st.session_state["prediction_quality_df"] = criteria_rank_df

            st.session_state["prediction_df"] = prediction_df
            st.session_state["prediction_descriptors_df"] = prediction_descriptor_df
            st.success(f"Prediction complete for {len(prediction_df)} peptides.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    prediction_df = st.session_state["prediction_df"]
    prediction_quality_df = st.session_state.get("prediction_quality_df")
    if isinstance(prediction_df, pd.DataFrame) and not prediction_df.empty:
        if "OptimizationDirection" in prediction_df.columns:
            direction = str(prediction_df["OptimizationDirection"].dropna().iloc[0]) if prediction_df["OptimizationDirection"].notna().any() else ""
            if direction == "minimize":
                st.info(
                    "This model target is interpreted as lower-is-better (for example IC50/MIC/DockingScore). "
                    "`Prediction` is shown as a positive optimized design score, while the original model output is kept in `RawModelPrediction`."
                )
        st.dataframe(prediction_df, use_container_width=True, height=300)
        _plot(
            plot_prediction_ranking(prediction_df, top_n=min(30, len(prediction_df))),
            key="predict_ranking_plot",
        )

        quality_source = (
            prediction_quality_df
            if isinstance(prediction_quality_df, pd.DataFrame) and not prediction_quality_df.empty
            else prediction_df
        )
        pass_cols = [c for c in quality_source.columns if c.startswith("Pass_")]
        if pass_cols and isinstance(quality_source, pd.DataFrame) and not quality_source.empty:
            st.markdown("#### Good Structure Criteria (Heuristic Screening)")
            quality_summary = summarize_structure_quality(quality_source)
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean Quality Score", f"{quality_summary.get('MeanStructureQualityScore', float('nan')):.2f}")
            c2.metric("Median Quality Score", f"{quality_summary.get('MedianStructureQualityScore', float('nan')):.2f}")
            c3.metric("High-Quality Peptides", quality_summary.get("HighQualityCount(score>=70)", 0))

            _plot(
                plot_structure_quality_distribution(quality_source),
                key="predict_structure_quality_distribution",
            )

            if "Rank" in quality_source.columns:
                rank_options = quality_source["Rank"].tolist()
                selected_rank = st.selectbox(
                    "Select ranked peptide for criteria radar",
                    options=rank_options,
                    index=0,
                    key="predict_structure_radar_rank",
                )
                selected_row = quality_source[quality_source["Rank"] == selected_rank].iloc[0]
            else:
                selected_idx = st.selectbox(
                    "Select peptide row for criteria radar",
                    options=list(range(len(quality_source))),
                    index=0,
                    key="predict_structure_radar_row",
                    format_func=lambda i: f"Row {i + 1}",
                )
                selected_row = quality_source.iloc[int(selected_idx)]
            _plot(
                plot_structure_criteria_radar(selected_row, pass_cols),
                key="predict_structure_radar",
            )
            with st.expander("Show criteria table for predicted peptides"):
                criteria_cols = ["Rank", "Name", "Sequence", "StructureQualityScore", "StructureAssessment"] + pass_cols
                criteria_cols = [c for c in criteria_cols if c in prediction_df.columns]
                if not criteria_cols:
                    criteria_cols = ["Name", "Sequence"] + pass_cols
                    criteria_cols = [c for c in criteria_cols if c in quality_source.columns]
                st.dataframe(quality_source[criteria_cols], use_container_width=True, height=280)

        pred_csv = dataframe_to_csv_bytes(prediction_df)
        pred_xlsx = dataframe_to_excel_bytes(
            {
                "Predictions": prediction_df,
                "PredictionDescriptors": st.session_state["prediction_descriptors_df"]
                if isinstance(st.session_state["prediction_descriptors_df"], pd.DataFrame)
                else pd.DataFrame(),
            }
        )
        d1, d2 = st.columns(2)
        d1.download_button(
            "Download Predictions (CSV)",
            data=pred_csv,
            file_name="peptide_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
        d2.download_button(
            "Download Predictions (Excel)",
            data=pred_xlsx,
            file_name="peptide_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


def _render_visualization_tab() -> None:
    st.subheader("6) Visualizations & Explainability")
    descriptor_df = st.session_state["descriptor_df"]
    descriptor_cols = st.session_state["descriptor_columns"]
    result = st.session_state["training_result"]
    prediction_df = st.session_state.get("prediction_df")

    if not isinstance(descriptor_df, pd.DataFrame) or descriptor_df.empty:
        st.info("Calculate descriptors first.")
        return

    st.markdown("#### 3D Peptide Viewer")
    sequence_options = descriptor_df["Sequence"].astype(str).head(300).tolist() if "Sequence" in descriptor_df.columns else []
    if sequence_options:
        selected_sequence = st.selectbox("Select peptide sequence for approximate 3D conformer", sequence_options, key="peptide_3d_sequence")
        if st.button("Generate 3D Peptide View", key="generate_peptide_3d_view"):
            with st.spinner("Generating approximate peptide 3D conformer..."):
                fig3d, status3d = _plot_peptide_3d(selected_sequence)
            st.caption(status3d)
            _plot(fig3d, key="peptide_3d_viewer")
    else:
        st.info("No peptide sequence column is available for 3D viewing.")

    st.markdown("#### Descriptor Landscape")
    c1, c2 = st.columns(2)
    with c1:
        _plot(
            plot_descriptor_distribution(descriptor_df, descriptor_cols, max_features=12),
            key="viz_descriptor_distribution",
        )
    with c2:
        _plot(
            plot_correlation_heatmap(descriptor_df, descriptor_cols, max_features=80),
            key="viz_descriptor_corr",
        )

    st.markdown("#### PCA / t-SNE / UMAP")
    emb_method = st.selectbox("Embedding method", ["PCA", "TSNE", "UMAP"])
    color_by = None
    potential_color_cols = [
        col for col in descriptor_df.columns if col not in descriptor_cols and col.lower() not in {"sequence"}
    ]
    if potential_color_cols:
        color_opts = ["(none)"] + potential_color_cols
        selected = st.selectbox("Color points by", color_opts, index=0)
        if selected != "(none)":
            color_by = descriptor_df[selected]

    if st.button("Compute Embedding"):
        try:
            with st.spinner(f"Running {emb_method} embedding..."):
                X_embed = descriptor_df[descriptor_cols].copy()
                if len(X_embed) > 1500:
                    X_embed = X_embed.sample(n=1500, random_state=42)
                    color_input = color_by.loc[X_embed.index] if color_by is not None else None
                else:
                    color_input = color_by
                emb = compute_embedding(X_embed, method=emb_method, random_state=42, n_components=2)
                st.session_state["embedding_df"] = (emb, color_input)
            st.success(f"{emb_method} embedding ready.")
        except Exception as exc:
            st.error(f"Embedding failed: {exc}")

    if st.session_state["embedding_df"] is not None:
        emb_df, color_input = st.session_state["embedding_df"]
        title = f"{emb_method} Embedding"
        fig = plot_embedding(emb_df, labels=color_input, title=title)
        _plot(fig, key="viz_embedding_plot")

    st.markdown("#### Detailed PCA Analysis")
    pca_components = st.slider("PCA components for analysis", min_value=2, max_value=20, value=8, step=1)
    if st.button("Run Detailed PCA"):
        try:
            with st.spinner("Computing PCA scores, variance profile, and loadings..."):
                X_pca = descriptor_df[descriptor_cols].copy()
                if len(X_pca) > 2500:
                    X_pca = X_pca.sample(n=2500, random_state=42)
                color_input = color_by.loc[X_pca.index] if color_by is not None else None
                pca_info = compute_pca_analysis(X_pca, random_state=42, n_components=pca_components)
                st.session_state["pca_analysis"] = (pca_info, color_input)
            st.success("Detailed PCA analysis ready.")
        except Exception as exc:
            st.error(f"PCA analysis failed: {exc}")

    if st.session_state["pca_analysis"] is not None:
        pca_info, pca_color = st.session_state["pca_analysis"]
        explained_df = pca_info["explained_variance"]
        scores_df = pca_info["scores"]
        loadings_df = pca_info["loadings"]

        _plot(plot_pca_explained_variance(explained_df), key="viz_pca_explained_variance")
        _plot(
            plot_embedding(
                scores_df[["PC1", "PC2"]],
                labels=pca_color,
                title="PCA Scores (PC1 vs PC2)",
            ),
            key="viz_pca_scores_scatter",
        )

        component_options = [c for c in loadings_df.columns if c.startswith("PC")]
        selected_component = st.selectbox(
            "Loading component",
            component_options,
            index=0,
            key="viz_pca_loading_component",
        )
        _plot(
            plot_pca_loading_bar(loadings_df, component=selected_component, top_n=20),
            key="viz_pca_loading_bar",
        )
        with st.expander("Show PCA loading table"):
            st.dataframe(loadings_df, use_container_width=True, height=260)

    if isinstance(result, dict):
        st.markdown("#### Model Comparison")
        metric_col = (
            "FitQuality_0_100"
            if result["task_type"] == "regression" and "FitQuality_0_100" in result["comparison_table"].columns
            else ("R2" if result["task_type"] == "regression" else "F1")
        )
        _plot(
            plot_model_comparison(result["comparison_table"], primary_metric=metric_col),
            key="viz_model_comparison",
        )

        model_name = st.selectbox("Model for explainability", list(result["model_outputs"].keys()), key="viz_model_picker")
        bundle = result["model_outputs"][model_name]["bundle"]

        native_importance = compute_model_feature_importance(bundle)
        if not native_importance.empty:
            _plot(
                plot_feature_importance(native_importance, top_n=25, title="Model Native Feature Importance"),
                key=f"viz_native_importance_{model_name}",
            )
            st.dataframe(native_importance.head(30), use_container_width=True, height=260)
        else:
            st.info("This model does not expose native feature importance.")

        st.markdown("#### SHAP Summary")
        if st.button("Compute SHAP Importance (may take time)"):
            try:
                with st.spinner("Computing SHAP values..."):
                    shap_df = compute_shap_importance(bundle, descriptor_df, max_samples=300)
                _plot(
                    plot_feature_importance(shap_df, top_n=25, title="SHAP Mean Absolute Contribution"),
                    key=f"viz_shap_importance_{model_name}",
                )
                st.dataframe(shap_df.head(30), use_container_width=True, height=260)
            except Exception as exc:
                st.warning(f"SHAP not available for this model/configuration: {exc}")

    if isinstance(prediction_df, pd.DataFrame) and not prediction_df.empty:
        pass_cols = [c for c in prediction_df.columns if c.startswith("Pass_")]
        if pass_cols:
            st.markdown("#### Predicted Peptide Structure Criteria")
            _plot(
                plot_structure_quality_distribution(prediction_df),
                key="viz_prediction_structure_distribution",
            )

    st.markdown("#### Position Design Suggestions")
    target_options = [
        col for col in descriptor_df.columns if col not in descriptor_cols and col.lower() not in {"sequence", "name"}
    ]
    numeric_targets = []
    for col in target_options:
        converted = pd.to_numeric(descriptor_df[col], errors="coerce")
        if converted.notna().sum() >= 3:
            numeric_targets.append(col)
    if not numeric_targets:
        st.info("Position suggestions require a numeric target such as Activity, BindingScore, DockingScore, IC50, or MIC.")
        return

    default_target = (
        st.session_state.get("target_column")
        if st.session_state.get("target_column") in numeric_targets
        else numeric_targets[0]
    )
    design_target = st.selectbox(
        "Target for position suggestions",
        numeric_targets,
        index=numeric_targets.index(default_target),
        key="design_target_col",
    )
    inferred_direction = infer_optimization_direction(design_target)
    direction_label = st.radio(
        "Optimization direction",
        ["auto", "maximize", "minimize"],
        index=0,
        horizontal=True,
        key="design_direction",
        help="Use minimize for IC50/MIC/docking scores where lower is better.",
    )
    min_count = st.slider("Minimum evidence count per residue-position", 1, 10, 2, key="design_min_count")

    if direction_label == "auto":
        st.caption(f"Auto direction for `{design_target}`: `{inferred_direction}`.")

    st.markdown("#### Best-Reference Sequence Alignment")
    if st.button("Analyze Best Peptide Alignment", key="run_best_reference_alignment"):
        try:
            alignment_df = sequence_alignment_to_reference(
                descriptor_df,
                sequence_col="Sequence",
                target_col=design_target,
                direction=direction_label,
            )
            profile_df = peptide_activity_design_profile(
                descriptor_df,
                target_col=design_target,
                descriptor_cols=descriptor_cols,
                direction=direction_label,
            )
            st.session_state["peptide_alignment_analysis"] = {
                "alignment": alignment_df,
                "profile": profile_df,
                "target": design_target,
            }
            st.success("Best-reference alignment and design profile are ready.")
        except Exception as exc:
            st.error(f"Best-reference alignment failed: {exc}")

    peptide_alignment = st.session_state.get("peptide_alignment_analysis")
    if isinstance(peptide_alignment, dict):
        alignment_df = peptide_alignment.get("alignment", pd.DataFrame())
        profile_df = peptide_alignment.get("profile", pd.DataFrame())
        if not alignment_df.empty:
            ref_name = alignment_df.iloc[0].get("ReferenceName", "Best reference")
            ref_seq = alignment_df.iloc[0].get("ReferenceSequence", "")
            st.caption(f"Reference peptide: {ref_name} | {ref_seq}")
            st.dataframe(alignment_df.head(50), use_container_width=True, height=320)
            if "Target" in alignment_df.columns:
                _plot(
                    px.scatter(
                        alignment_df,
                        x="EditSimilarityToReference",
                        y="Target",
                        hover_name="Name" if "Name" in alignment_df.columns else None,
                        template="plotly_white",
                        title="Sequence similarity to best peptide vs target",
                    ),
                    key="design_best_reference_similarity",
                )
        if not profile_df.empty:
            st.markdown("#### Best-Activity Structure Criteria")
            st.dataframe(profile_df, use_container_width=True, height=260)

    if st.button("Analyze Position Effects", key="run_position_design"):
        try:
            analysis = positional_activity_analysis(
                descriptor_df,
                sequence_col="Sequence",
                target_col=design_target,
                direction=direction_label,
                min_count=min_count,
                max_positions=60,
            )
            st.session_state["design_analysis"] = analysis
            st.success("Position-level design analysis is ready.")
        except Exception as exc:
            st.error(f"Position design analysis failed: {exc}")

    design_analysis = st.session_state.get("design_analysis")
    if isinstance(design_analysis, dict):
        effects_df = design_analysis.get("position_residue_effects", pd.DataFrame())
        rec_df = design_analysis.get("position_recommendations", pd.DataFrame())
        class_df = design_analysis.get("class_recommendations", pd.DataFrame())

        if not effects_df.empty:
            _plot(plot_position_residue_heatmap(effects_df), key="design_position_heatmap")
        if not rec_df.empty:
            _plot(plot_position_recommendation_scores(rec_df), key="design_position_recommendations_plot")
            st.dataframe(rec_df, use_container_width=True, height=300)
        if not class_df.empty:
            with st.expander("Residue class suggestions by position"):
                st.dataframe(class_df, use_container_width=True, height=260)

        if isinstance(prediction_df, pd.DataFrame) and not prediction_df.empty and not rec_df.empty:
            st.markdown("#### Mutation Ideas For New Peptides")
            candidate_options = prediction_df["Sequence"].astype(str).tolist()
            candidate_sequence = st.selectbox(
                "Select predicted/new peptide",
                candidate_options,
                index=0,
                key="design_candidate_sequence",
            )
            mutation_df = suggest_mutations_for_sequence(candidate_sequence, rec_df, top_n=15)
            if not mutation_df.empty:
                st.dataframe(mutation_df, use_container_width=True, height=300)


def _render_export_tab() -> None:
    st.subheader("7) Export Report")
    validated_df = st.session_state["validated_df"]
    descriptor_df = st.session_state["descriptor_df"]
    result = st.session_state["training_result"]
    prediction_df = st.session_state["prediction_df"]
    prediction_quality_df = st.session_state.get("prediction_quality_df")

    if not isinstance(validated_df, pd.DataFrame) or validated_df.empty:
        st.info("No validated input data found yet.")
        return

    input_summary = {
        "Total Valid Peptides": len(validated_df),
        "Invalid Sequences": len(st.session_state["invalid_df"]) if isinstance(st.session_state["invalid_df"], pd.DataFrame) else 0,
        "Duplicate Sequences": len(st.session_state["duplicate_df"]) if isinstance(st.session_state["duplicate_df"], pd.DataFrame) else 0,
        "Task Type": TASK_VALUE_TO_DISPLAY.get(st.session_state["task_type"], st.session_state["task_type"]),
    }

    descriptor_summary = {
        "Descriptors Computed": isinstance(descriptor_df, pd.DataFrame) and not descriptor_df.empty,
        "Descriptor Rows": len(descriptor_df) if isinstance(descriptor_df, pd.DataFrame) else 0,
        "Descriptor Feature Count": len(st.session_state["descriptor_columns"]),
        "Descriptor Config": describe_descriptor_set(st.session_state["descriptor_config"]),
    }

    model_summary: dict[str, Any] = {"Model Trained": False}
    metrics_summary: dict[str, Any] = {}
    comparison_df = None
    figures: dict[str, Any] = {}

    if isinstance(result, dict):
        best = result["best_model_name"]
        model_summary = {
            "Model Trained": True,
            "Best Model": best,
            "Models Compared": len(result["model_outputs"]),
            "Training Dataset Size": result.get("dataset_size", "N/A"),
            "Target Column": result["model_outputs"][best]["bundle"].target_column,
        }
        metrics_summary = result["model_outputs"][best]["test"]["metrics"]
        comparison_df = result["comparison_table"]
        metric_col = (
            "FitQuality_0_100"
            if result["task_type"] == "regression" and "FitQuality_0_100" in comparison_df.columns
            else ("R2" if result["task_type"] == "regression" else "F1")
        )
        figures["Model Comparison"] = plot_model_comparison(comparison_df, primary_metric=metric_col)

    if isinstance(descriptor_df, pd.DataFrame) and not descriptor_df.empty:
        figures["Descriptor Distribution"] = plot_descriptor_distribution(
            descriptor_df,
            st.session_state["descriptor_columns"],
            max_features=10,
        )
        figures["Descriptor Correlation"] = plot_correlation_heatmap(
            descriptor_df,
            st.session_state["descriptor_columns"],
            max_features=50,
        )
    if isinstance(prediction_df, pd.DataFrame) and not prediction_df.empty:
        figures["Prediction Ranking"] = plot_prediction_ranking(prediction_df, top_n=min(30, len(prediction_df)))
        if "StructureQualityScore" in prediction_df.columns:
            figures["Structure Quality Distribution"] = plot_structure_quality_distribution(prediction_df)

    if st.button("Generate HTML Report", type="primary"):
        report_html = generate_html_report(
            input_summary=input_summary,
            descriptor_summary=descriptor_summary,
            model_summary=model_summary,
            metrics=metrics_summary,
            prediction_df=prediction_df if isinstance(prediction_df, pd.DataFrame) else None,
            comparison_df=comparison_df if isinstance(comparison_df, pd.DataFrame) else None,
            figures=figures,
        )
        st.session_state["last_report_html"] = report_html
        st.success("HTML report generated.")

    if st.session_state["last_report_html"]:
        st.download_button(
            "Download HTML Report",
            data=st.session_state["last_report_html"].encode("utf-8"),
            file_name=f"peptide_qsar_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True,
        )

    export_tables: dict[str, pd.DataFrame] = {
        "ValidatedInput": validated_df,
    }
    if isinstance(descriptor_df, pd.DataFrame) and not descriptor_df.empty:
        export_tables["Descriptors"] = descriptor_df
    if isinstance(prediction_df, pd.DataFrame) and not prediction_df.empty:
        export_tables["Predictions"] = prediction_df
    if isinstance(result, dict):
        export_tables["ModelComparison"] = result["comparison_table"]
    if isinstance(prediction_quality_df, pd.DataFrame) and not prediction_quality_df.empty:
        export_tables["StructureCriteria"] = prediction_quality_df

    workbook_bytes = dataframe_to_excel_bytes(export_tables)
    st.download_button(
        "Download Full Workbook (Excel)",
        data=workbook_bytes,
        file_name=f"peptide_qsar_exports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def _render_about_tab() -> None:
    st.subheader("About / Documentation")
    st.markdown("### Developed by Ahmed G. Soliman")
    st.dataframe(pd.DataFrame(DEVELOPER_PROFILE.items(), columns=["Item", "Details"]), use_container_width=True, hide_index=True)
    st.link_button("Open developer portfolio", DEVELOPER_PORTFOLIO, use_container_width=True)

    st.markdown(
        """
### Peptide QSAR Prediction Tool

This local Windows-friendly application supports peptide sequence upload, peptide descriptor calculation,
QSAR model training, activity/property prediction, model evaluation, explainability, PCA analysis,
best-reference sequence alignment, position-level design suggestions, and report export.

### Scientific and Software Basis

- **Peptide/protein descriptors:** Biopython `ProteinAnalysis` / ProtParam-style descriptors were used for
  molecular weight, aromaticity, instability index, GRAVY, isoelectric point, net charge, secondary-structure
  fractions, and extinction coefficients.
- **Custom peptide descriptors:** amino acid composition, dipeptide composition, residue class frequencies,
  elemental composition, hydrophobic moment, charge density, Chou-Fasman helix/sheet propensities, Boman-style
  binding/solubility scale, motif fingerprints, and optional approximate 3D conformer shape descriptors.
- **Machine learning:** scikit-learn estimators are used for regression, binary classification, and multiclass
  classification, including linear models, SVR/SVM, kNN, Random Forest, Extra Trees, Gradient Boosting, Naive
  Bayes, and MLP models.
- **Visualization:** Plotly is used for interactive descriptor distributions, heatmaps, PCA scores, loading
  plots, model comparison, prediction ranking, confusion matrices, and design-suggestion heatmaps.
- **Design guidance:** best-reference peptide alignment, edit similarity, position-wise residue effects,
  candidate mutation suggestions, and best-activity descriptor windows are estimated from the uploaded dataset.
- **3D handling:** RDKit can build approximate peptide conformers from one-letter sequences with `MolFromFASTA`
  and ETKDG embedding for visual inspection and 3D shape descriptors. This is a computational approximation,
  not an experimentally determined structure.
- **Explainability:** model-native feature importance is used where available; SHAP support is optional when
  installed.
- **Windows packaging:** Streamlit is launched locally through `run_app.bat`; PyInstaller can build a launcher
  or the single-file launcher included with this project.

### Documentation Sources Used

- Streamlit documentation for app layout, tabs, file upload, and Plotly rendering:
  https://docs.streamlit.io/
- Biopython ProtParam / `ProteinAnalysis` documentation:
  https://biopython.org/wiki/ProtParam
- scikit-learn user guide and estimator documentation:
  https://scikit-learn.org/stable/user_guide.html
- pandas file I/O documentation:
  https://pandas.pydata.org/docs/
- Plotly Python documentation:
  https://plotly.com/python/
- SHAP documentation:
  https://shap.readthedocs.io/
- PyInstaller documentation:
  https://pyinstaller.org/en/stable/

### Developer Statement

This application was developed by Ahmed G. Soliman for peptide QSAR modeling, peptide descriptor engineering,
activity prediction, design guidance, 3D peptide visualization, and scientific reporting.
        """
    )

    manuscript_md_path = DOCS_DIR / "Peptide_QSAR_Tool_Manuscript_Methods.md"
    manuscript_pdf_path = DOCS_DIR / "Peptide_QSAR_Tool_Manuscript_Methods.pdf"
    if manuscript_pdf_path.exists():
        st.download_button(
            "Download manuscript-ready scientific documentation (PDF)",
            data=manuscript_pdf_path.read_bytes(),
            file_name=manuscript_pdf_path.name,
            mime="application/pdf",
            use_container_width=True,
        )
    if manuscript_md_path.exists():
        st.download_button(
            "Download editable manuscript documentation (Markdown)",
            data=manuscript_md_path.read_bytes(),
            file_name=manuscript_md_path.name,
            mime="text/markdown",
            use_container_width=True,
        )

    pdf_path = DOCS_DIR / "Peptide_QSAR_Tool_Documentation.pdf"
    if pdf_path.exists():
        st.download_button(
            "Download quick PDF documentation",
            data=pdf_path.read_bytes(),
            file_name=pdf_path.name,
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("PDF documentation file is not present yet. It can be generated from the project docs folder.")


def main() -> None:
    _init_session_state()

    st.sidebar.title("Peptide QSAR Tool")
    st.sidebar.write("Local, beginner-friendly peptide QSAR workflow")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
**Tabs**
1. Home  
2. Upload Data  
3. Descriptors  
4. Train Model  
5. Evaluate  
6. Predict  
7. Visualizations  
8. Export Report
9. About
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: Start with the example dataset if you are new.")

    tabs = st.tabs(
        [
            "Home",
            "Upload Data",
            "Descriptors",
            "Train Model",
            "Evaluate",
            "Predict New Peptides",
            "Visualizations",
            "Export Report",
            "About",
        ]
    )

    with tabs[0]:
        _render_home_tab()
    with tabs[1]:
        _render_upload_tab()
    with tabs[2]:
        _render_descriptors_tab()
    with tabs[3]:
        _render_train_tab()
    with tabs[4]:
        _render_evaluate_tab()
    with tabs[5]:
        _render_predict_tab()
    with tabs[6]:
        _render_visualization_tab()
    with tabs[7]:
        _render_export_tab()
    with tabs[8]:
        _render_about_tab()


if __name__ == "__main__":
    main()
