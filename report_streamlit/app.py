import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os
import pyvista as pv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.Factory.MaskRebuilder.Rebuilder import global_feature_collection, global_clustering, compute_components_distribution, get_new_masks
from src.Factory.load import load_pair_from_ct, build_records

# ================= CONFIG =================
st.set_page_config(
    page_title="Fetal Ossification Analysis",
    layout="wide"
)

# ================= CACHE =================
@st.cache_data
def load_data():
    df = pd.read_csv("report_streamlit/data/features.csv")
    cross = pd.read_csv("report_streamlit/data/cross_norm.csv", index_col=0)

    df["cluster"] = df["cluster"].astype(int)
    df["label"] = df["label"].astype(int)
    df["log_size"] = np.log1p(df["size"])

    return df, cross


df, cross = load_data()
# ================= THEME =================
theme = st.sidebar.toggle("Dark mode", True)
template = "plotly_dark" if theme else "plotly_white"

# ================= SIDEBAR =================
page = st.sidebar.radio(
    "Navigation",
    [
        "1. Introduction",
        "2. Dataset",
        "3. Data Engineering",
        "4. Feature Analysis",
        "5. Clustering",
        "6. Validation",
        "7. 3D Visualization",
        "8. Conclusion"
    ]
)
if page == "1. Introduction":

    st.title("1. Fetal Ossification Centers Analysis")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total components", len(df))
    col2.metric("Original classes", df["label"].nunique())
    col3.metric("Clusters", df["cluster"].nunique())
    col4.metric("Avg size", int(df["size"].mean()))
    reduction = 1 - (df["cluster"].nunique() / df["label"].nunique())
    col4.metric("Reduction", f"{int(reduction * 100)}%")

    st.markdown("""
    ### Research Context and Problem Definition

    The analyzed problem concerns the processing of medical data characterized by high complexity and limited availability. The CT images used in this study capture the development of fetal skeletal structures, making the dataset particularly rare and difficult to obtain.

    This is primarily due to the nature of the analyzed cases. Imaging of fetal skeletal development is associated with significant ethical and clinical constraints. In practice, this means that computed tomography in pregnant patients is performed extremely rarely and only in specific diagnostic situations, such as severe trauma or acute conditions where other imaging modalities are insufficient.

    CT relies on ionizing radiation, which interacts with matter at the atomic level by ejecting electrons from atoms and molecules. This process leads to the formation of ions and reactive chemical species, which can damage biological structures such as proteins, cell membranes, and particularly DNA. The early prenatal period is especially sensitive, as cells undergo rapid division and differentiation, making them highly susceptible to external factors, including radiation.

    DNA damage may result in mutations, disruption of cell division, and consequently an increased risk of cancer. In the case of a developing fetus, this risk is particularly significant due to the high rate of cellular proliferation. Furthermore, prenatal exposure to ionizing radiation may lead to developmental abnormalities, depending on the dose and the stage of pregnancy. For these reasons, the use of CT imaging during pregnancy is strictly limited to clinically justified cases.

    As a result, the number of available scans is very limited, and their acquisition requires strict medical and formal procedures. The available cases are often non-standard and frequently associated with fetal pathologies.

    Access to such data is further constrained by ethical requirements, including the need for appropriate approvals and data anonymization. In practice, obtaining these datasets is possible mainly through collaboration with medical institutions. Despite these challenges, such data are highly valuable, as they enable the development of methods aimed at improving the detection and understanding of potential abnormalities during pregnancy.

    At the same time, the problem itself is inherently challenging from a computational perspective. The structures of interest — ossification centers — are extremely small, typically occupying around 0.01% of the CT volume, with irregular shapes and high variability across different stages of fetal development. Their appearance changes across slices, and the contrast between structures can be subtle.

    Combined with the limited and biased nature of the dataset, this makes reliable segmentation particularly difficult and pushes the problem beyond standard medical image segmentation tasks.
    """)



    st.markdown(
        "**This context directly motivates the design of the data engineering and modeling pipeline presented below.**")

    st.markdown("---")

    st.markdown("""
    ### From Raw Data to Anatomical Structure Representation
    
    A key objective of this project was not only to analyze the data, but to reconstruct meaningful anatomical representations from raw volumetric inputs.

    The original dataset does not provide directly usable structures. Instead, segmentation masks are encoded in compressed formats and must be decoded and reconstructed at the voxel level. This process enables the identification of individual anatomical components within each CT volume.

    Each detected component is then transformed into a feature representation, capturing its spatial location and size. This abstraction allows the problem to shift from voxel-level analysis to object-level reasoning.

    Clustering is subsequently applied to group these components into a reduced set of structures. Importantly, this process is not evaluated purely through numerical metrics, but through spatial reconstruction.

    By projecting cluster assignments back into the voxel space, it is possible to visually verify that the resulting structures remain spatially coherent. This demonstrates that the clustering process preserves meaningful anatomical organization, despite a significant reduction in label complexity.
    
    This project addresses the challenge of analyzing fetal CT data, where anatomical structures are not directly accessible but must be reconstructed from compressed representations.

    The objective is not limited to segmentation. Instead, the pipeline focuses on transforming raw volumetric data into a structured representation of anatomical components, enabling higher-level reasoning about their spatial organization.

    The workflow integrates:
    - custom data engineering for decoding medical formats,
    - feature-based representation of anatomical structures,
    - unsupervised learning to identify structural patterns.

    This approach shifts the problem from voxel-level processing to object-level analysis.
    """)

    st.markdown("---")

    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total components", len(df))
    col2.metric("Original labels", df["label"].nunique())
    col3.metric("Clusters", df["cluster"].nunique())
    col4.metric("Avg component size", int(df["size"].mean()))

    st.markdown("---")

    st.subheader("Project Overview")

    fig = px.scatter(
        df.sample(min(2000, len(df))),
        x="z",
        y="y",
        color="cluster",
        template=template,
        title="Spatial distribution of anatomical components"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    The visualization above illustrates the spatial distribution of anatomical components after feature extraction.

    Each point represents an individual structure detected in the CT volumes.  
    The clustering reveals that components are not randomly distributed, but form coherent spatial groups.

    This observation motivates the use of unsupervised learning as a tool for structural analysis rather than purely predictive modeling.
    """)
elif page == "2. Dataset":

    st.title("2. Dataset and Data Complexity")

    st.markdown("""
        ### Nature of the Data

        The dataset consists of volumetric CT scans of fetal pelvic regions, accompanied by segmentation masks describing ossification centers.

        Unlike standard machine learning datasets, the data is not provided in a structured format. Instead, it is distributed across multiple files, requiring reconstruction before it can be used.

        Key challenges include:
        - absence of a unified tabular structure,
        - compressed mask representations (RLE),
        - indirect relationships between CT volumes and segmentation masks.
        """)

    st.markdown("---")

    st.subheader("Dataset Characteristics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total components", len(df))
    col2.metric("Unique anatomical labels", df["label"].nunique())
    col3.metric("Avg components per scan", int(df.groupby("case_id").size().mean()))

    st.markdown("---")

    st.subheader("Distribution of Anatomical Components")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x="log_size",
            nbins=50,
            template=template,
            title="Distribution of component sizes (log scale)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df.groupby("case_id").size().reset_index(name="count"),
            x="count",
            nbins=30,
            template=template,
            title="Number of components per CT scan"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        ### Interpretation

        The distribution of component sizes reveals a strong imbalance between small and large anatomical structures.  
        This is characteristic of medical imaging data, where fine-grained structures coexist with larger regions.

        Additionally, the number of components per scan varies significantly, indicating heterogeneity across cases.

        These observations highlight the necessity of a dedicated data engineering pipeline before applying machine learning methods.
        """)
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Total samples", df["case_id"].nunique())
    col2.metric("Total components", len(df))
    col3.metric("Avg components / CT", int(df.groupby("case_id").size().mean()))
    col4.metric("Max components / CT", int(df.groupby("case_id").size().max()))
    st.subheader("Sample of extracted features")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Mean HU (avg)", int(df["hu_mean"].mean()))
    col2.metric("Std HU (avg)", int(df["hu_std"].mean()))
    col3.metric("Min HU", int(df["hu_min"].min()))
    col4.metric("Max HU", int(df["hu_max"].max()))
    st.dataframe(
        df[["case_id", "label", "cluster", "size", "z", "y", "x"]].head(50),
        use_container_width=True
    )
    st.markdown("""
    ### The dataset exhibits high structural variability across samples.
    The number of anatomical components varies significantly between CT scans,
    indicating heterogeneity in the underlying anatomical structures.
    Additionally, the distribution of component sizes is highly imbalanced,
    with many small structures and fewer large ones.
    This complexity makes the dataset challenging and requires a dedicated
    data engineering pipeline before any modeling step.""")
    st.markdown("---")
    st.subheader("CT Intensity Scale (Hounsfield Units)")
    st.markdown("""
    CT images are represented using **Hounsfield Units (HU)**, a standardized scale that quantifies the radiodensity of tissues based on X-ray attenuation.

    Unlike typical image representations (e.g. RGB or grayscale values), HU values have a direct physical interpretation:

    - Air: ~ -1000 HU  
    - Water: 0 HU  
    - Soft tissue: ~ 30–80 HU  
    - Bone: 300–1000+ HU  

    This means that each voxel value encodes how much the tissue attenuates X-ray radiation, making the data inherently quantitative rather than purely visual.

    As a result, preprocessing must preserve the physical meaning of these values, while normalization is required for stable model training.

    This dual nature — physical interpretability and numerical variability — adds another layer of complexity to the dataset.
    """)
    fig = px.histogram(
        df,
        x="hu_mean",
        nbins=40,
        template=template,
        title="Distribution of mean CT intensities"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Interpretation of Intensity Values

    The observed intensity values significantly exceed the standard Hounsfield Unit range typically used in CT imaging.

    This suggests that the data is stored in a transformed or non-normalized format, rather than directly representing calibrated HU values.

    Such behavior is common in raw medical datasets, where intensity scaling or offsets must be accounted for during preprocessing.

    Understanding this discrepancy is essential, as it directly affects how intensity values are interpreted and normalized for further analysis.
    """)
    st.subheader("Why Data Engineering is Required")

    st.markdown("""
        The dataset cannot be directly used for training due to its structure.

        Before any modeling step, it is necessary to:
        - reconstruct volumetric data from raw files,
        - decode segmentation masks from compressed formats,
        - align CT scans with corresponding annotations.

        These operations form the foundation of the pipeline and are described in the next section.
        """)
    # ============================ Data Engineering ================================
elif page == "3. Data Engineering":

    st.title("3. Data Engineering Pipeline")

    st.markdown("""
        ### From Unstructured Medical Files to Usable Data

        The most challenging aspect of this project was not model training,
        but reconstructing usable data from heterogeneous medical files.

        The dataset was not provided in a structured format. Instead, it consisted of multiple interdependent files,
        requiring reverse engineering to recover relationships between CT volumes and segmentation masks.

        Without this step, the dataset is unusable for any machine learning task.
        """)

    st.markdown("---")
    st.subheader("Data Structure Complexity")

    st.markdown("""
        The data is distributed across multiple file types:

        - `.job3` → list of samples
        - `.arterydata` → metadata and file references
        - `.rdata` → raw CT volumes
        - `.maskSet` → compressed segmentation masks
        - `.header` → volume dimensions

        Relationships between these files are not explicit and must be reconstructed.
        """)

    st.code("""
    DATA_1:
    [path/to/ct.rdata]

    MASK_1_0:
    [mask description]
    """, language="text")

    st.markdown("""
        The lack of a unified structure requires custom parsing logic to extract relevant information.
        """)

    st.markdown("---")
    st.subheader("Decoding Compressed Masks (RLE)")

    st.markdown("""
        Segmentation masks are stored using Run-Length Encoding (RLE), a compressed format that stores lengths of pixel sequences.

        Instead of storing full voxel grids, the data contains sequences of runs:
        - alternating between background and foreground
        - requiring reconstruction at voxel level
        """)

    st.code("""
    runs = seg[2:-2]
    val = 1

    for length in runs:
        if val == 1:
            label_vec[idx:idx+length] = label
        idx += length
        val ^= 1
    """, language="python")

    st.markdown("""
        The decoding process reconstructs full 3D segmentation masks.

        A key constraint is that the sum of runs must match the total number of voxels,
        ensuring data consistency.
        """)

    st.markdown("---")
    st.subheader("Reconstruction Pipeline")

    st.markdown("""
        The complete data engineering pipeline can be summarized as follows:
        """)

    st.code("""
    RAW FILES
       ↓
    .job3 parsing
       ↓
    arterydata extraction
       ↓
    file selection (done / done2)
       ↓
    RLE decoding (maskSet)
       ↓
    volume reconstruction
       ↓
    CT + MASK aligned volumes
    """, language="text")

    st.markdown("""
        The output of this process is a consistent dataset of aligned 3D volumes,
        ready for further analysis and modeling.

        This transformation is essential and represents the foundation of the entire system.
        """)

elif page == "4. Feature Analysis":

    st.title("4. Feature Engineering and Structural Analysis")

    st.markdown("""
        ### From Voxels to Anatomical Structures

        This stage goes beyond standard feature extraction.

        Instead of operating directly on voxel-level data, segmentation masks are decomposed into individual anatomical components.
        Each component is treated as an object and described using geometric and spatial features.

        This transformation enables a shift from low-level image processing to structural analysis.
        """)

    st.markdown("---")

    # ================= FEATURES =================

    st.subheader("Component Representation")

    st.markdown("""
        For each connected component, the following features are extracted:

        - **Centroid (normalized)** → spatial position within the volume  
        - **Size** → number of voxels (volume proxy)  
        - **Bounding box** → spatial extent  
        - **Dominant label** → anatomical class  

        These features provide a compact and interpretable representation of anatomical structures.
        """)

    st.markdown("---")

    # ================= HISTOGRAM =================

    st.subheader("Distribution of Component Sizes")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x="log_size",
            nbins=50,
            template=template,
            title="Log-transformed size distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df,
            x="cluster",
            y="log_size",
            template=template,
            title="Cluster vs Size Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        ### Interpretation
        Distribution of Component Sizes

        The histogram reveals the distribution of component sizes after logarithmic transformation.

        Despite normalization, the distribution remains moderately right-skewed, indicating a persistent imbalance between small and large anatomical structures. This is consistent with the nature of medical imaging data, where numerous fine-grained structures coexist with fewer large regions.

        Histograms are particularly effective in revealing the shape and skewness of data distributions, making them a fundamental tool in exploratory data analysis :contentReference[oaicite:0]{index=0}.

        From a modeling perspective, this size imbalance is critical. Without transformation, clustering algorithms such as K-Means would be dominated by large components due to their magnitude, leading to biased grouping.

        The boxplot complements this analysis by providing a compact summary of distributions across clusters, highlighting variability, spread, and potential outliers :contentReference[oaicite:1]{index=1}.

        The visualization shows that while variability exists, distributions across clusters remain relatively consistent. This suggests that clustering is not driven purely by size, but by a combination of spatial and structural features.

        Importantly, this indicates that the feature engineering pipeline preserves meaningful anatomical variability rather than introducing artificial grouping.
    
        """)

    st.markdown("---")

    # ================= SPATIAL =================

    st.subheader("Spatial Distribution of Components")

    fig = px.scatter(
        df.sample(min(3000, len(df))),
        x="z",
        y="y",
        color="cluster",
        template=template,
        title="Component distribution in spatial domain"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        The spatial distribution demonstrates that anatomical components are not randomly located.

        Instead, they form structured regions in space, which validates the use of centroid-based features for further analysis.
        """)

    st.markdown("---")

    # ================= TABLE =================

    st.subheader("Feature Table Sample")

    feature_df = df.copy()
    if "centroid_norm" in feature_df.columns:
        if isinstance(feature_df["centroid_norm"].iloc[0], (list, tuple, np.ndarray)):
            feature_df["z_norm"] = feature_df["centroid_norm"].apply(lambda x: x[0])
            feature_df["y_norm"] = feature_df["centroid_norm"].apply(lambda x: x[1])
            feature_df["x_norm"] = feature_df["centroid_norm"].apply(lambda x: x[2])
        else:
            # już rozbite
            feature_df.rename(columns={
                "z": "z_norm",
                "y": "y_norm",
                "x": "x_norm"
            }, inplace=True)

    # bbox — tylko jeśli to lista
    if "bbox" in feature_df.columns:
        if isinstance(feature_df["bbox"].iloc[0], (list, tuple, np.ndarray)):
            feature_df["bbox_z"] = feature_df["bbox"].apply(lambda x: x[0])
            feature_df["bbox_y"] = feature_df["bbox"].apply(lambda x: x[1])
            feature_df["bbox_x"] = feature_df["bbox"].apply(lambda x: x[2])

    # compactness (jeśli bbox istnieje)
    if all(col in feature_df.columns for col in ["bbox_z", "bbox_y", "bbox_x"]):
        feature_df["compactness"] = (
                                            feature_df["bbox_z"] * feature_df["bbox_y"] * feature_df["bbox_x"]
                                    ) / (feature_df["size"] + 1e-6)
    st.dataframe(
        feature_df[
            [
                "cluster",
                "size",
                "z_norm",
                "y_norm",
                "x_norm",
                "bbox_z",
                "bbox_y",
                "bbox_x",
                "compactness"
            ]
        ].head(50),
        use_container_width=True)

    st.markdown("""
        The table above illustrates the structured representation of anatomical components.

        Each row corresponds to a single detected structure, enabling analysis at the object level instead of voxel level.
        """)
    st.markdown("""The table illustrates the transformed representation of anatomical components.

        Instead of raw voxel data, each structure is described using geometric and spatial features.
        This enables analysis at the object level and forms the basis for clustering.

        Such representation significantly reduces complexity while preserving essential structural information.
        """)

elif page == "5. Clustering":

    st.title("5. Clustering and Structural Grouping")

    st.markdown("""
    ### From Feature Space to Anatomical Organization

    After transforming voxel-level data into structured representations,
    clustering is applied to group anatomical components based on spatial and geometric similarity.

    The objective is not only dimensionality reduction, but the discovery of meaningful structural organization.
    """)

    st.markdown("---")

    # ================= SELECT =================

    clusters = st.multiselect(
        "Select clusters",
        sorted(df["cluster"].unique()),
        default=sorted(df["cluster"].unique())
    )

    filtered = df[df["cluster"].isin(clusters)]

    # ================= 2D =================

    st.subheader("Spatial Distribution (2D Projection)")

    fig = px.scatter(
        filtered,
        x="z",
        y="y",
        color="cluster",
        template=template,
        title="Cluster distribution in spatial domain"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    The 2D projection clearly reveals that clusters are spatially localized rather than randomly distributed.

    Components assigned to the same cluster occupy consistent regions within the anatomical space, indicating that the feature representation encodes meaningful positional information.

    This behavior confirms that centroid-based features are sufficient to capture global anatomical structure.
    """)

    st.markdown("")

    st.markdown("---")

    # ================= 3D =================

    st.subheader("3D Structural Representation")

    fig = px.scatter_3d(
        filtered,
        x="z",
        y="y",
        z="x",
        color="cluster",
        size="log_size",
        opacity=0.7,
        template=template
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    The 3D visualization provides a direct view of the anatomical organization preserved by clustering.

    Despite reducing the number of labels, the global structure of ossification centers remains intact.

    Clusters form volumetric regions that correspond to consistent anatomical zones, demonstrating that the transformation from voxel space to feature space retains spatial coherence.
    """)

    st.markdown("")

    st.markdown("---")

    # ================= HEATMAP =================

    st.subheader("Cluster–Label Relationship")

    fig = px.density_heatmap(
        x=df["cluster"],
        y=df["label"],
        nbinsx=22,
        nbinsy=55,
        color_continuous_scale="viridis",
        title="Mapping between clusters and original labels"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    The mapping between clusters and original labels reveals that multiple anatomical classes are grouped into higher-level structures.

    This reduction does not eliminate information, but reorganizes it into a more compact and interpretable representation.

    The clustering process therefore acts as a structural abstraction layer rather than a simple compression mechanism.
    """)

    st.markdown("")

    st.markdown("---")

    st.subheader("Cluster vs Ground Truth Comparison")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.scatter(
            df,
            x="z",
            y="y",
            color="cluster",
            template=template,
            title="Clusters (Model Representation)"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            df,
            x="z",
            y="y",
            color="label",
            template=template,
            title="Original Labels (Ground Truth)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    The comparison between clustered representation and original labels highlights a critical result.

    Although the number of classes is significantly reduced, the spatial arrangement of structures remains highly consistent.

    This demonstrates that clustering preserves anatomical organization while simplifying the representation, which is essential for downstream modeling.
    """)
    st.markdown("""
    This result validates the entire pipeline, from data engineering to feature representation and clustering.
    """)



elif page == "6. Validation":
    st.title("6. Validation and Reconstruction")

    st.markdown("""
    ### Validation in Original Voxel Space

    The previous stages operate on transformed representations of the data.
    However, validation must be performed directly on the original CT volumes.

    This section verifies whether the clustering results remain consistent
    when projected back into voxel space.

    The goal is to ensure that structural coherence is preserved in real anatomical data.
    """)

    st.markdown("---")
# ================= 2D VALIDATION =================
elif page == "6. Validation":
    st.write(" VALIDATION PAGE ACTIVE")
    st.title("6. Validation and Reconstruction")

    st.markdown("""
    ### Validation in Original Voxel Space

    This section verifies whether clustering results remain consistent
    when projected back into voxel space.
    """)



    import plotly.graph_objects as go

    st.subheader("2D Slice Validation")

    df_raw = pd.read_csv('report_streamlit/data/principal_sharp_only.csv')

    @st.cache_resource
    def compute_pipeline(df):
        test_records = build_records(df)
        all_features = global_feature_collection(test_records)
        all_features = global_clustering(
            all_features,
            compute_components_distribution(test_records)
        )
        return get_new_masks(all_features, test_records)

    if "result" not in st.session_state:
        st.session_state.result = compute_pipeline(df_raw)

    result = st.session_state.result

    col1, col2 = st.columns([2, 1])

    with col1:
        sample = st.selectbox("Sample", [r[0] for r in result])

    with col2:
        mode = st.radio("Mask", ["Before", "After"])

    selected = next(r for r in result if r[0] == sample)
    _, ct, mask_before, new_mask = selected

    if mode == "Before":
        mask = mask_before.copy()
    else:
        mask = new_mask.copy()

    opacity = st.slider("Opacity", 0.0, 1.0, 0.5)

    if ct.ndim != 3:
        st.error(f"Invalid CT shape: {ct.shape}")
        st.stop()

    Z = ct.shape[0]
    z_idx = st.slider("Slice", 0, Z - 1, Z // 2)

    img = ct[z_idx].astype(float)
    img = np.nan_to_num(img)

    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255).astype(np.uint8)

    mask_slice = (mask[z_idx] > 0).astype(np.uint8) * 255

    if img.ndim != 2 or mask_slice.ndim != 2:
        st.error("Invalid slice dimensionality")
        st.stop()

    img = img[::2, ::2]
    mask_slice = mask_slice[::2, ::2]

    fig = go.Figure()

    img_rgb = np.stack([img, img, img], axis=-1)
    fig.add_trace(go.Image(z=img_rgb))

    mask_rgb = np.zeros((*mask_slice.shape, 3), dtype=np.uint8)
    mask_rgb[..., 0] = mask_slice

    fig.add_trace(go.Image(z=mask_rgb, opacity=opacity))

    fig.update_layout(height=750, margin=dict(l=0, r=0, t=20, b=0))

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
elif page == "7. 3D Visualization":

    st.subheader("3D Structural Validation")
    st.markdown("""
    The 3D reconstruction provides a global view of anatomical structures.

    Surfaces extracted from CT volumes and segmentation masks allow evaluation
    of spatial consistency across the entire volume.

    This step is essential, as local correctness in 2D slices does not guarantee
    global structural coherence.
    """)
    df_raw = pd.read_csv('report_streamlit/data/principal_sharp_only.csv')

    @st.cache_resource
    def compute_pipeline(df):
        test_records = build_records(df)
        all_features = global_feature_collection(test_records)
        all_features = global_clustering(
            all_features,
            compute_components_distribution(test_records)
        )
        return get_new_masks(all_features, test_records)

    if "result_3d" not in st.session_state:
        st.session_state.result_3d = compute_pipeline(df_raw)

    result = st.session_state.result_3d

    samples = [r[0] for r in result]
    sample = st.selectbox("Select sample", samples, key="3d")

    selected = next(r for r in result if r[0] == sample)
    _, ct, mask_before, new_mask = selected

    ct = ct.copy()
    mask_before = mask_before.copy()
    mask_after = new_mask.copy()

    ct = ct[::3, ::3, ::3]
    mask_before = mask_before[::3, ::3, ::3]
    mask_after = mask_after[::3, ::3, ::3]

    ct = np.nan_to_num(ct)
    ct = ct.astype(np.float32)

    vmin, vmax = np.percentile(ct, [5, 95])
    ct = np.clip(ct, vmin, vmax)
    ct = (ct - vmin) / (vmax - vmin + 1e-6)

    grid = pv.wrap(ct)
    grid_before = pv.wrap((mask_before > 0).astype(np.uint8))
    grid_after = pv.wrap((mask_after > 0).astype(np.uint8))

    default_val = 0.5

    if default_val < ct.min() or default_val > ct.max():
        default_val = float(np.percentile(ct, 70))

    threshold = st.slider(
        "Bone threshold",
        float(ct.min()),
        float(ct.max()),
        default_val
    )

    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("black")

    ct_surface = grid.contour([threshold])
    show_before = st.toggle("Show BEFORE", True)
    show_after = st.toggle("Show AFTER", True)
    if ct_surface.n_points > 0:
        plotter.add_mesh(ct_surface, color="white", opacity=0.2)

    if show_before:
        before_surface = grid_before.contour([0.5])
        if before_surface.n_points > 0:
            plotter.add_mesh(before_surface, color="green", opacity=0.5)

    if show_after:
        after_surface = grid_after.contour([0.5])
        if after_surface.n_points > 0:
            plotter.add_mesh(after_surface, color="red", opacity=0.5)

    angle = st.slider("Rotate", 0, 360, 45)

    plotter.reset_camera()
    plotter.camera.zoom(1.3)
    plotter.camera.Azimuth(angle)
    plotter.enable_eye_dome_lighting()

    plotter.show(auto_close=False)

    img = plotter.screenshot()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, use_container_width=True)
    st.markdown("""### 3D Visualization in Medical Imaging

3D visualization constitutes one of the most critical components of modern medical diagnostics, particularly in the context of volumetric imaging such as CT scans. Continuous advancements in this domain directly contribute to improving diagnostic precision and, ultimately, patient outcomes. Developing and refining methods for medical image analysis is therefore not only a technical challenge, but also a meaningful contribution to healthcare and life preservation.

The 3D reconstruction presented above should be interpreted as an approximation of the original volumetric data. It relies on surface extraction techniques that capture only selected structural characteristics of the scan. Consequently, the generated geometry reflects only certain traits of the underlying volume and remains a simplified representation rather than a fully faithful reconstruction.

This highlights the inherent gap between lightweight, web-based visualization approaches and dedicated medical imaging environments. Specialized tools such as *napari* operate directly on voxel-level data, enabling precise, high-fidelity inspection without intermediate approximation.

To emphasize this difference, the following images were generated from the same dataset using napari. They illustrate the true level of detail and structural accuracy achievable when working directly with volumetric medical data.
""")
    def show_image(path, caption):
        st.markdown(f"""
            <div style="height: 350px; display: flex; align-items: center; justify-content: center;">
                <img src="{path}" style="max-height: 100%; max-width: 100%; object-fit: contain;">
            </div>
            <p style="text-align: center; font-size: 14px;">{caption}</p>
        """, unsafe_allow_html=True)

    colx,coly,colz = st.columns([1, 2, 1])
    with colx:
        show_image("report_streamlit/assets/ct1_napari.png", caption="Clustered and Reconstructed Structures CT with mask")
    with coly:
        show_image("report_streamlit/assets/ct3_maskGT.png",caption="Clustered and reconstructed mask")
    with colz:
        show_image("report_streamlit/assets/ct2_napari.png", caption="Grand Truth of the same example")


elif page == "8. Conclusion":
    st.title("Conclusion")
    st.markdown("""
        ### 
        
This project demonstrates how complex volumetric medical data can be transformed into a structured representation suitable for machine learning.

Through a combination of data engineering, feature extraction, and clustering, it is possible to significantly reduce label complexity while preserving anatomical coherence.

The results confirm that:

- meaningful structures can be reconstructed from raw encoded data,
- clustering captures spatial organization rather than trivial properties,
- dimensionality reduction does not necessarily imply loss of structural information.

Beyond its technical aspects, this work highlights the broader impact of data-driven approaches in medical imaging.

Advances in this field contribute directly to improving diagnostic tools and understanding anatomical structures, ultimately supporting better clinical outcomes.
        """)