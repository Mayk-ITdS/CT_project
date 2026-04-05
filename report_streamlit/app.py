from pathlib import Path
import os
import sys
from PIL import Image
from cyclopts.help.formatters import markdown
from docutils.nodes import caption
from matplotlib import pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0,ROOT)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ================= CONFIG =================
st.set_page_config(
    page_title="Fetal Ossification Analysis",
    layout="wide"
)
ROOT_DIRECTORY = Path(__file__).parent.parent.resolve()
HERE = ROOT_DIRECTORY / 'report_streamlit'
# ================= CACHE =================
@st.cache_data
def load_data():
    df = pd.read_csv("report_streamlit/data/features.csv")
    cross = pd.read_csv("report_streamlit/data/cross_norm.csv", index_col=0)
    df_raw = pd.read_csv("report_streamlit/data/eda_full.csv")
    df["cluster"] = df["cluster"].astype(int)
    df["label"] = df["label"].astype(int)
    df["log_size"] = np.log1p(df["size"])

    return df, cross, df_raw

def clamp_format(x):
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.4f}".rstrip("0").rstrip(".")

df, cross, df_raw = load_data()
# ================= THEME =================
theme = st.sidebar.toggle("Dark mode", True)
template = "plotly_dark" if theme else "plotly_white"

# ================= SIDEBAR =================
page = st.sidebar.radio(
    "Report Sections",
    [
        "1. Problem & Dataset",
        "2. Exploratory Data Analysis",
        "3. Data Preparation",
        "4. Model Training",
        "5. Evaluation & Results",
        "6. Conclusion & Future Work"
    ]
)
if page == "1. Problem & Dataset":

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

    st.markdown("---")

    st.markdown("""
    ### From Raw Data to Anatomical Structure Representation

    The dataset used in this study consists of volumetric CT scans of fetal pelvic regions, accompanied by segmentation masks describing ossification centers.

    In total:
    - 89 CT volumes are available (based on eda_full.csv)
    - Each case includes:
      - a CT scan (in .rdata format)
      - a corresponding segmentation mask
      - in some cases, multiple variants of the same scan (e.g., sharp vs blurred)
    
    This introduces an additional level of complexity, as multiple representations of the same anatomical structure must be considered during data selection.
    
    Unlike standard machine learning datasets, the data is not provided in a structured format.
    Instead, it is distributed as a compressed archive containing multiple folders and heterogeneous file types.
    
    The dataset includes:
    - raw volumetric data (.rdata)
    - segmentation masks (.maskSet / .voxelizedSurfaces)
    - metadata files (.job3, .arterydata)
    - header files describing volume geometry
    
    Importantly, these files are not directly linked in a consistent or explicit manner.
    
    As a result, the first step of the analysis is not exploratory visualization,
    but reconstruction of the dataset itself.
    
    This involves:
    - identifying valid CT–mask pairs,
    - resolving inconsistencies between multiple data sources,
    - decoding compressed segmentation formats,
    - and aligning volumetric data into consistent representations.
    
    Only after this reconstruction step a meaningful exploratory data analysis shall be performed.
    
    Additionally, the original dataset contains 55 distinct segmentation labels,
    which are not directly aligned with anatomical structures and exhibit inconsistencies across samples.
    
    This further motivates the need for structural analysis and label simplification,
    introduced later in the pipeline.

    """)

    st.markdown("---")

    st.subheader("Dataset Summary")

    summary_df = pd.DataFrame({
        "Metric": [
            "Number of CT volumes",
            "Average mask coverage",
            "Number of distinct classes ( initially )",
            "Min CT intensity",
            "Max CT intensity",
            "Mean CT intensity"
        ],
        "Value": [
            df_raw["sample_nr"].nunique(),
            round(df_raw["coverage"].mean(), 4),
            int(df["label"].nunique()),
            int(df_raw["ct_min"].min()),
            int(df_raw["ct_max"].max()),
            int(df_raw["ct_mean"].mean())
        ]
    })
    summary_df['Value'] = summary_df['Value'].map(clamp_format)

    st.table(summary_df)

    col1, col2, col3, col4 = st.columns(4)

    st.markdown("---")

    st.markdown("""
        The dataset cannot be directly used for training due to its structure.

        Before any modeling step, it is necessary to:
        - reconstruct volumetric data from raw files,
        - decode segmentation masks from compressed formats,
        - align CT scans with corresponding annotations.

        These operations form the foundation of the pipeline and are described in the next section.
        """)
    # ============================ Data Engineering ================================
elif page == '2. Exploratory Data Analysis':
    st.title("2.1 Data Reverse Engineering Pipeline")

    st.markdown("""
        ### From Unstructured Medical Files to Usable Data
        Between raw data exploration and dataset construction,
        an intermediate stage is required, which can be described as **reverse engineering.**
        
        **This was in fact the most challenging aspect of this project.**
       
        At this stage, the dataset is not yet defined in a usable form.
        Instead, it exists as a collection of heterogeneous files with implicit and undocumented relationships.

        Reverse engineering is therefore necessary to:
        - identify valid data sources,
        - reconstruct relationships between CT volumes and segmentation masks,
        - recover structural information from metadata,
        - and transform raw files into a consistent dataset representation.
        **Without this step, the dataset is unusable for any machine learning task.**
        """)

    st.markdown("---")
    st.subheader("Data Structure Complexity/Variety and Data Types")

    st.markdown("""
        The data is distributed across multiple file types:

        - `.job3` → list of samples
        - `.arterydata` → metadata and file references
        - `.rdata` → raw CT volumes
        - `.maskSet` → compressed segmentation masks
        - `.header` → volume dimensions

        Relationships between these files are not explicit and must be reconstructed.
        """)
    st.image('report_streamlit/assets/path_files_chaos.png',width='content',caption='Unlike standard medical datasets (e.g., DICOM or NIfTI),this dataset does not provide a unified representation, requiring reverse engineering before any analysis can be performed')
    st.markdown("""
    The raw dataset is distributed as a collection of loosely structured file paths,
    without explicit relationships between CT volumes and segmentation masks.

    The left panel shows a .job3 file listing sample-specific metadata paths,
    while the right panel illustrates the internal structure of an .arterydata file.
    
    Notably, the dataset exhibits several inconsistencies:
    - multiple alternative data sources (e.g., done vs done2),
    - lack of explicit pairing between CT volumes and masks,
    - indirect references requiring additional parsing,
    - inconsistent naming conventions.
    
    This structure prevents direct loading of the dataset and necessitates
    a dedicated reconstruction pipeline. 
    
    
    """)
    st.caption("""
        To resolve these inconsistencies, a heuristic file selection strategy is introduced,
        which dynamically evaluates multiple candidate sources and selects the first valid one.
        """)
    st.code("""
    def pick_done_file(sample_path: Path) -> tuple[Path, list[str] | None]:

    folder = sample_path.parent
    candidates = [
        folder / "done.arterydata",
        folder / "done2.arterydata",
        sample_path,
    ]

    uniq = []
    for c in candidates:
        if c not in uniq:
            uniq.append(c)

    for c in uniq:
        lines = read_lines_if_exists(c)
        if has_mask_key(lines):
            return c, lines

    return sample_path, read_lines_if_exists(sample_path)
    """, language="python")

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
    st.write("""
            The encoded sequence contains a header and footer, which must be excluded, cleaned.
            The remaining values represent alternating runs of foreground and background voxels,
            requiring iterative reconstruction.
            A key constraint is that the sum of runs must match the total number of voxels,
            ensuring data consistency.
            """,unsafe_allow_html=True)
    st.code("""
    runs = seg[2:-2]
    val = 1

    for length in runs:
        if val == 1:
            label_vec[idx:idx+length] = label
        idx += length
        val ^= 1
    """, language="python")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            The decoding process reconstructs full 3D segmentation masks.

            """,text_alignment='left',width='content')

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
            The output of the reconstruction pipeline is a dataset of aligned 3D CT volumes
            and corresponding segmentation masks, forming the effective ground truth used
            for subsequent analysis and modeling.
            
            Unlike standard medical datasets, where annotations are provided in a structured
            and directly usable format, the ground truth in this study is the result of an
            explicit reconstruction process.
            
            The figure below illustrates this transformation: from raw CT data, through
            decoded segmentation masks, to spatially consistent 3D representations of
            individual anatomical structures.
            
            This representation enables direct volumetric analysis and serves as the basis
            for all downstream tasks.
            """)
    col1,col2,col3 = st.columns(3)

    with col1:
        st.image('report_streamlit/assets/corps1.png',
                 caption='CT + mask (colors)')
    with col2:
        st.image('report_streamlit/assets/corps2.png',caption='Reconstructed CT')
    with col3:
        st.image('report_streamlit/assets/corps3.png',caption='Mask Reconstructed')
    st.caption("""
    Figure: Reconstructed ground truth obtained from the data engineering pipeline.
    Top: CT volume with overlaid segmentation structures.
    Middle: raw CT slice.
    Bottom: extracted segmentation components visualized in 3D space.

    The visualization demonstrates that anatomical structures are not explicitly
    encoded in the raw data, but emerge only after reconstruction and alignment.
    """)
    st.markdown("""
    It is important to note that this reconstructed ground truth is not perfect.
    The masks originate from heterogeneous sources and may contain inconsistencies,
    noise, and class ambiguities.

    Therefore, the reconstructed dataset should be interpreted as an operational
    ground truth, rather than a clinically validated reference and nnow is ready for 
    Exploratory Data Analysis.
    """)
    st.markdown("---")
    st.markdown("""
    The reconstructed volumes provide the first opportunity to analyze
    the spatial and structural properties of the dataset.
    
    Figure X reveals that ossification centers are:
    - small relative to the full volume,
    - spatially sparse,
    - and highly fragmented.
    
    These observations motivate a detailed exploratory data analysis,
    focusing on spatial distribution, intensity characteristics,
    and class imbalance.
    """)
    st.markdown("---")

    def parse_shape(s):
        s = s.strip("()")
        return list(map(int, s.split(',')))

    df_raw[['Z', 'Y', 'X']] = df_raw['ct_shape'].apply(parse_shape).apply(pd.Series)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df_raw,
            x="coverage",
            nbins=40,
            template=template,
            title="Mask Coverage",
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        The distribution of mask coverage reveals that anatomical structures
        occupy only a very small fraction of the volumetric space, typically below 2%.
        This indicates an extreme voxel-level class imbalance, where the vast majority
        of voxels correspond to background.
        
        Such imbalance renders standard metrics such as accuracy uninformative,
        as trivial background predictions would achieve high scores without detecting
        any relevant structures. Therefore, overlap-based metrics such as the Dice coefficient
        are required to properly evaluate segmentation performance.
        
        Moreover, the low coverage highlights the geometric sparsity of the problem.
        Ossification centers are small, spatially dispersed, and fragmented,
        making their detection significantly more challenging.
        
        These characteristics directly influence the design of the data processing pipeline
        and justify the use of ROI-based sampling and specialized loss functions
        adapted to highly imbalanced segmentation tasks.
        """)
        st.markdown("---")
    with col2:
        fig = px.histogram(
            df_raw,
            x="ct_mean",
            nbins=40,
            template=template,
            title="CT Mean Intensity",
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        The distribution of mean CT intensities reveals significant variability across samples,
        indicating that the dataset is not standardized in terms of intensity scale.
        
        Moreover, the observed values do not correspond to typical Hounsfield Units,
        suggesting that the data is stored in a transformed or preprocessed representation.
        As a result, absolute intensity values cannot be directly interpreted
        in a clinical context.
        
        This variability reduces comparability between samples and introduces
        an additional source of complexity for the learning process.
        Without appropriate normalization, the model would be prone to learning
        scan-specific intensity patterns rather than generalizable anatomical features.
        
        Therefore, per-sample normalization is required to standardize the input space
        and enable robust learning across heterogeneous data.
        """)
        st.markdown("---")
    corr = df_raw[['coverage', 'ct_min', 'ct_max', 'ct_mean', 'Z', 'Y', 'X']].corr()
    col_left, col_center, col_right = st.columns([0.5, 3, 0.5])

    with col_center:
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="viridis",
            aspect="auto",
            template=template,
            title="Feature Correlation"
        )

        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)
    col_left1, col_center1, col_right1 = st.columns([0.5, 3, 0.5])
    with col_center1:
        st.markdown("""
        The correlation analysis reveals meaningful relationships between structural
        and intensity-based features.
        
        A strong positive correlation between mask coverage and maximum intensity
        confirms that segmented regions correspond to high-density anatomical structures,
        consistent with ossification centers.
        
        In contrast, a strong negative correlation between minimum and maximum intensity
        indicates significant variability in intensity scaling across samples,
        suggesting heterogeneous acquisition or preprocessing conditions.
        
        Additionally, the weak correlation between volumetric depth and other variables
        indicates that anatomical content is largely independent of the number of slices,
        which justifies the use of ROI-based approaches instead of full-volume processing.
        
        Overall, these relationships highlight the complex interplay between intensity,
        geometry, and structure in the dataset, and further support the design choices
        made in the data processing pipeline.
        """)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(
            df_raw,
            x="Z",
            nbins=40,
            template=template,
            title="Volume Depth (Z)",
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        The distribution of volumetric depth reveals substantial variability across samples,
        with the number of slices ranging widely and lacking standardization.
        
        This variability prevents direct processing of full volumes within a unified model,
        as consistent input dimensions are required for training.
        
        Moreover, the lack of correlation between depth and anatomical content indicates
        that larger volumes do not necessarily contain more relevant information.
        
        These observations justify the use of ROI-based sampling and fixed-size sub-volumes,
        which enable efficient and consistent processing of heterogeneous volumetric data.
        
        """)

    with col4:
        fig = px.scatter(
            df_raw,
            x="coverage",
            y="ct_max",
            template=template,
            title="Coverage vs CT Max",
            opacity=0.7
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        The relationship between mask coverage and maximum CT intensity
        reveals a clear positive correlation, indicating that larger segmented regions
        are associated with higher intensity values.
        
        This is consistent with the physical properties of CT imaging,
        where ossified structures exhibit high density and therefore high intensity values.
        
        The observed relationship validates the alignment between segmentation masks
        and underlying anatomical structures, confirming that the dataset captures
        meaningful physical characteristics.
        
        At the same time, the variability in the relationship suggests heterogeneity
        in structure size and acquisition conditions, indicating that intensity alone
        is insufficient for reliable segmentation and must be complemented by spatial context.
        """)
    st.markdown("---")
    # 5. Summary table
    summary = df_raw[['coverage', 'ct_min', 'ct_max', 'ct_mean', 'Z', 'Y', 'X']].describe()

    st.table(summary)
    st.markdown('---')
    st.markdown("""
    The exploratory data analysis reveals several key challenges inherent to the dataset.

    First, anatomical structures occupy only a small fraction of the volumetric space,
    resulting in extreme voxel-level class imbalance.
    
    Second, significant variability in volumetric dimensions prevents direct processing
    of full volumes in a unified framework.
    
    Third, intensity distributions are not standardized across samples,
    introducing additional complexity for model generalization.
    
    Finally, anatomical structures are highly fragmented and spatially sparse,
    forming multiple disconnected components rather than continuous regions.
    
    These observations indicate that direct voxel-level learning is suboptimal.
    Instead, a transformation of the data representation is required.
    
    This motivates the transition from voxel-level processing to
    structure-aware data preparation, described in the next section.
    
    """)
if page == '3. Data Preparation':
    st.title("From Structural Analysis to feature engineering")

    st.markdown("""
    ### Structural Inconsistency of Labels and Clustering Approach

    Exploratory analysis revealed that the original segmentation labels
    are not consistently aligned with anatomical structures.
    
    In particular, spatial visualizations suggested that multiple labels
    often correspond to the same anatomical region, forming coherent
    spatial clusters despite having different identifiers.
    
    This indicates that the original labeling scheme is fragmented
    and does not reflect the true anatomical organization.
    
    Since no predefined mapping between labels was available,
    a data-driven approach was adopted.
    
    Each connected component was treated as an individual structure
    and represented using:
    
    - normalized centroid (spatial position),
    - size (volume proxy).
    
    These features were used to perform clustering using the K-Means algorithm.
    
    The number of clusters was estimated based on the median number
    of components per volume, providing a data-adaptive approximation
    of anatomical complexity.
    
    The resulting clusters preserved the spatial organization of structures,
    while reducing label fragmentation.
    
    This transformation effectively redefines the segmentation space,
    grouping geometrically consistent components into higher-level
    structural entities.
        """)

    st.markdown("---")
    # ================= FEATURES =================
    st.title("Clustering and Structural Grouping")
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

    st.markdown("---")

    st.title("Region of Interest Extraction and Slicing")

    st.markdown("""
    ### From Structural Representation to Model Input

    While clustering resolves inconsistencies in label representation,
    it does not address the geometric challenges identified during exploratory analysis.

    In particular, anatomical structures remain:
    - small relative to the full volume,
    - spatially sparse,
    - and embedded within large background regions.

    As a result, direct processing of full CT volumes remains inefficient and suboptimal.

    To address this, a region-of-interest (ROI) based sampling strategy is introduced.
    """)

    st.markdown("---")

    st.subheader("ROI Extraction")

    st.markdown("""
    For each slice, connected components are identified and processed individually.

    The following steps are applied:

    - detection of connected components,
    - computation of bounding boxes,
    - expansion of regions using a margin,
    - clamping to valid image boundaries,
    - extraction of local patches.

    This ensures that each region contains relevant anatomical information
    while excluding unnecessary background.
    """)

    st.markdown("---")

    st.subheader("Slicing Strategy (2.5D Representation)")

    st.markdown("""
    Instead of using full 3D volumes, a 2.5D approach is employed.

    For each central slice:
    - adjacent slices are stacked,
    - forming a multi-channel input representation.

    This provides local volumetric context while maintaining computational efficiency.

    The final input has the form:
    (5, 128, 128)

    where:
    - 5 represents adjacent slices,
    - 128×128 is the normalized spatial resolution.
    """)

    st.markdown("---")

    st.subheader("Normalization and Filtering")

    st.markdown("""
    To ensure consistency across samples:

    - intensity values are normalized per volume,
    - low-information regions are filtered based on mask coverage,
    - duplicate samples are removed.

    These steps reduce noise and improve training stability.
    """)

    st.markdown("---")
    st.markdown("""
    ### Final Training Dataset Construction

    Through the combination of clustering, region-of-interest extraction,
    and slicing, the original volumetric data is transformed into
    a structured training dataset.
    
    Each CT volume, after passing through the full pipeline:
    - structural decomposition,
    - semantic grouping (clustering),
    - spatial localization (ROI),
    - and slicing into fixed-size inputs,
    
    produces a collection of standardized samples.
    
    These samples are stored as independent training instances,
    forming the final dataset used for model training.
    
    This transformation effectively converts a small number of heterogeneous
    volumetric scans into a large and consistent dataset of localized
    training examples.
    
    The resulting dataset is summarized in the metadata table,
    which captures properties of each generated sample,
    including spatial location, coverage, and structural characteristics.
    
    This representation enables:
    
    - dataset-level analysis,
    - reproducibility,
    - and controlled sampling strategies during training.
    """)
    col1, col2, col3, col4 = st.columns(4)
    meta = pd.read_csv('report_streamlit/meta_index_full.csv')
    col1.metric("Total samples", len(meta))
    col2.metric("Unique cases", meta["case_id"].nunique())
    col3.metric("Avg coverage", f"{meta['real_fill'].mean():.3f}")
    col4.metric("Median coverage", f"{meta['real_fill'].median():.3f}")
    st.subheader("Generated Training Dataset")

    st.dataframe(
        meta.head(50),
        use_container_width=True
    )
    st.markdown("""
    Each row in the table corresponds to a single training sample
    generated through the preprocessing pipeline.
    
    The dataset no longer represents full CT volumes,
    but localized regions containing anatomical structures.
    
    This transformation significantly increases the number of training instances,
    while preserving structural information.
    
    As a result, the learning process operates on a derived dataset,
    which is specifically designed to address the challenges identified
    during exploratory data analysis.
    
    """)
elif page == "4. Model Training":

    st.title("4. Model Training")
    st.markdown("""
    ### Problem Formulation

    The task is formulated as a multi-class semantic segmentation problem,
    where each pixel is assigned to one of the predefined ossification classes
    or background.
    
    However, the primary challenge does not lie in class discrimination,
    but in detecting the presence of anatomical structures within highly sparse data.
    
    The dataset is characterized by:
    
    - a limited number of original volumetric scans,
    - a significantly expanded number of training samples through slicing,
    - and an extreme class imbalance at the pixel level.
    
    Although slicing increases the number of samples,
    it does not resolve the imbalance problem,
    as the vast majority of pixels still correspond to background.
    
    At this stage, the class space is already simplified through clustering,
    allowing the model to operate on a more consistent and anatomically meaningful
    set of labels.
    
    ### Final Definition of Anatomical Classes

    During the later stage of the project,
    additional information regarding the true anatomical labeling
    of the dataset became available.
    
    It was established that the segmentation task involves
    a set of seven anatomically defined classes,
    corresponding to specific ossification centers.
    
    This clarification significantly improved the consistency
    and interpretability of the problem.
    
    In contrast to the initial dataset, which contained up to 55 fragmented labels,
    the final class definition provides a coherent and clinically meaningful
    representation of anatomical structures.
    
    As a result, the modeling task can be formulated using a well-defined
    and significantly simplified label space.
    """)
    st.markdown("""
    Importantly, the model does not operate independently.

    The Dataset, sampling strategy, and model architecture jointly define
    the effective learning system.

    The dataset determines what information is available,
    the sampler controls how it is presented,
    and the model defines how it is interpreted.

    This interaction is critical in highly imbalanced medical segmentation tasks.
    """)
    st.markdown('---')
    st.markdown("""
    ### Training Pipeline and Data Sampling

    The training process operates on the dataset constructed during preprocessing,
    represented as a collection of localized samples.
    
    A custom dataset class is used to load:
    - multi-channel input tensors (2.5D slices),
    - corresponding segmentation masks.
    
    To address the imbalance in sample difficulty,
    a difficulty-aware sampling strategy is introduced.
    
    Samples are categorized based on mask coverage,
    which reflects the proportion of foreground pixels:
    
    - easy samples → high coverage,
    - medium samples → moderate coverage,
    - hard samples → low coverage.
    
    This categorization allows the training process to control
    the distribution of samples seen by the model.
    
    A custom sampler is then used to construct batches
    that balance these difficulty levels,
    preventing the model from being dominated by trivial background examples.
    """)
    st.markdown("---")
    st.markdown("### Mathematical Formulation")

    st.latex(r"""
    f_{\theta}: \mathbb{R}^{C \times H \times W} \rightarrow \mathbb{R}^{K \times H \times W}
    """)

    st.markdown("""
    where:

    - C = number of input channels (2.5D slices)  
    - H, W = spatial dimensions  
    - K = number of classes  

    The model predicts a probability distribution for each pixel.
    """)
    st.markdown("""
        #### Choice of Loss Function

        The model was trained using a combination of
        Custom loss function combining 
        **Cross-Entropy Loss** and **Dice coefficient**
        As well as a Tversky loss function. 
        
        ##### Cross-Entropy
        """)
    st.latex(r"""
        \mathcal{L}_{CE} = - \sum_{i} y_i \log(p_i)
        """)
    st.markdown("""
        - provides stable gradients  
        - optimizes pixel-wise classification  
        - ensures global convergence  

        However, it is strongly biased toward dominant classes.
        """)
    st.markdown("""
        ##### Dice Metric
                """)
    st.latex(r"""
    Dice = \frac{2TP}{2TP + FP + FN}
    """)
    st.markdown("""
        Dice coefficient measures the overlap between prediction and ground truth.
    
        It is particularly suitable for segmentation tasks with imbalanced classes,
        as it directly evaluates the quality of detected structures.
    
        Unlike accuracy, Dice is sensitive to:
        - false negatives,
        - false positives,
        - spatial overlap.
    
        This makes it a more meaningful metric for medical image segmentation.
        
        """)

    st.markdown("""
        ##### Tversky Loss
        """)
    st.latex(r"""
    T = \frac{TP}{TP + \alpha FP + \beta FN}
    """)

    st.latex(r"""
    \mathcal{L}_{Tversky} = 1 - T
        """)
    st.markdown("""
        The Tversky index introduces asymmetric penalization of errors.
        In this work:
        ** α = 0.3 β = 0.7**
        which gives us:  
        - higher penalty for false negatives  
        - improved sensitivity to small structures  

        This is critical in medical applications, where missing a structure
        is more severe than over-segmentation.

        #### Combined Objective

        The final loss balances:

        - global stability (Cross-Entropy)  
        - sensitivity to rare structures (Tversky)  
        """)

    st.markdown("---")
    st.markdown("""
    ### Why CNN
    
    CNN donc -> Convolutional Neural Networks were selected due to their ability to capture
    **spatial dependencies** between neighbouring pixels and local patterns
    in contrast to classical machine learning models such as Random Forests or MLPs,
    CNN preserve the spatial arrangement of pixels and exploit local context through convolutional filters.


    This is very important in medical imaging, where the meaning of a pixel, which defines 
    anatomical structures, depends not only
    on its intensity but also on its sorroundings.
    
    The use of convolution enables:
    - translation-invariant feature extraction,
    - detection of local patterns,
    - hierarchical representation learning.
        These characteristics make CNNs particularly well-suited for segmentation tasks involving
        small, irregular, and spatially dependent anatomical structures.
    """)

    st.markdown("---")

    st.markdown("""
    ### Model Architecture

    A **2.5D U-Net architecture with a ResNet34 encoder** was employed.
    
    The input consists of 5 adjacent slices, allowing the model to incorporate
    limited volumetric context without the computational cost of full 3D CNNs.
    The model is based on a U-Net architecture with a ResNet34 encoder.

    The encoder processes the input through a hierarchy of feature maps with increasing depth:
    64 → 128 → 256 → 512 channels, enabling progressive abstraction of spatial features.
    
    Unlike standard implementations, the model operates on a **5-channel input**, corresponding
    to a 2.5D representation of adjacent CT slices.
    
    The decoder consists of five convolution layers, each performing:
    
    - upsampling
    - feature fusion via skip connections,
    - convolutional refinement.
    A  key distinction between U-Net and conventional convolutional architectures is
    **Skip connections** feature which concatenates encoder and decoder feature-layers, preserving spatial detail 
    by directly passing high-resolution features
    from the encoder to the decoder,
    that would otherwise be lost during downsampling. 
    
    The final segmentation head maps the feature representation to a 7-class output
    using a convolutional layer.
    
    The U-Net structure enables:
    - multi-scale feature extraction,
    - preservation of spatial detail via skip connections,
    - accurate localization of small anatomical structures.
    
    The use of 2.5D input increases the receptive field along the depth dimension
    without requiring full 3D convolution This provides additional contextual information 
    while optimising computational efficiency, while the ResNet encoder improves gradient flow 
    and stabilizes training through residual connections.
    """)
    st.image("report_streamlit/assets/my_unet_diagram.png",width=700)
    st.caption("U-Net architecture (adapted from Ronneberger et al., 2015)")
    st.markdown("---")

    st.markdown("""
    ### Training Strategy

    The model was trained from scratch, without pre-trained weights.

    This decision was motivated by the strong domain mismatch between natural images
    and CT data, particularly in terms of intensity distribution and anatomical structure.
    
    Training from scratch allows the model to learn representations
    specific to fetal ossification patterns.
    
    The training process operates on a dataset constructed through the preprocessing pipeline,
    where each sample corresponds to a localized region of interest represented as a
    multi-channel input of shape (5, 128, 128).
    
    To address the extreme class imbalance and sparsity of anatomical structures,
    a difficulty-aware sampling strategy was introduced.
    
    Samples are categorized based on mask coverage, which reflects the proportion
    of foreground pixels within each sample. This results in three groups:
    
    easy samples (high coverage),
    medium samples,
    hard samples (low coverage, highly sparse structures).
    
    A custom sampler is used to construct batches that balance these difficulty levels,
    ensuring that informative samples are consistently presented to the model during training.
    
    This approach does not eliminate class imbalance at the pixel level,
    but mitigates its impact by controlling the distribution of samples seen by the model.
    
    Importantly, the training process does not operate on raw volumetric data,
    but on a transformed representation generated by the preprocessing pipeline.
    As a result, the dataset, sampling strategy, and model jointly define
    the effective learning space of the system.
    """)

    st.markdown("---")



    st.markdown("---")


elif page == "5. Evaluation & Results":

    st.title("5. Evaluation & Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Train Dice", "0.69")
    col2.metric("Test Dice", "0.72")
    col3.metric("Samples", "19,980")

    st.markdown("---")

    st.subheader("Per-case Dice Distribution")

    df_case = pd.read_csv("report_streamlit/data/test_per_case.csv")

    fig = px.histogram(
        df_case,
        x="dice",
        nbins=50,
        template=template
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("""
    Distribution of Dice scores across individual samples.
    The histogram illustrates the variability of segmentation performance
    at the case level, capturing both successful predictions and challenging cases.
    """)
    st.markdown("""
    The distribution reveals a wide spread of Dice scores, indicating
    significant variability in segmentation performance across samples.
    
    Notably, a substantial portion of cases achieves Dice scores approaching 0.7,
    which indicates a strong overlap between predictions and ground truth,
    especially given the extreme sparsity and complexity of the task.
    
    This level of performance confirms that the model successfully captures
    meaningful anatomical patterns and is capable of reliable localization
    of ossification centers.
    
    At the same time, the presence of a tail of low-performing samples
    reflects the inherent difficulty of certain cases, likely associated
    with small structure size, low contrast, or anatomical variability.
    
    These results suggest that while the model already achieves solid performance,
    further improvements are feasible through fine-tuning, enhanced sampling strategies,
    or architectural refinements.
    
    Importantly, the absence of a sharp peak suggests that the model does not
    overfit to a narrow subset of cases, but instead generalizes across varying conditions.
    
    Achieving Dice scores at this level in a highly imbalanced medical segmentation task
    indicates that the proposed pipeline is effective and well-aligned with the problem characteristics.
    """)

    st.markdown("---")

    st.subheader("Per-class Performance")

    df_class = pd.read_csv("report_streamlit/data/test_per_class.csv")

    fig = px.bar(
        df_class,
        x="class",
        y="dice",
        template=template
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    Per-class Dice performance across anatomical structures.
    The bar chart shows segmentation quality for each class,
    highlighting differences in model performance depending on structure characteristics.
    """)
    st.markdown("""
    Performance varies across classes, reflecting differences in structure size,
    contrast, and frequency within the dataset.
    
    Classes corresponding to larger or more distinct anatomical structures
    achieve higher Dice scores, while smaller or less defined structures
    remain more challenging.
    
    This confirms that segmentation performance is strongly influenced
    by geometric and spatial properties of the targets, rather than
    pure classification difficulty.
    
    The results further support the need for specialized handling
    of small structures in medical segmentation tasks.
    """)
    st.markdown("---")

    st.subheader("Difficulty Distribution")

    meta = pd.read_csv("report_streamlit/data/meta_index_full.csv")

    fig = px.histogram(
        meta,
        x="real_fill",
        nbins=50,
        template=template,
        labels={"real_fill": "Real Fill"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    The dataset is dominated by difficult samples, confirming the challenging nature of the task.
    """)

    st.markdown("---")

    st.subheader("Key Insight")

    st.markdown("""
    The distribution is skewed toward low-coverage samples,
    indicating that the dataset is dominated by difficult cases.
    
    This confirms the extreme sparsity of anatomical structures,
    where most samples contain only a minimal amount of foreground information.
    
    Such distribution directly explains the observed performance variability
    and highlights the importance of difficulty-aware sampling strategies
    introduced during training.
    
    The prevalence of hard samples reinforces that the problem is inherently
    detection-driven rather than classification-driven.
    
    Overall, the results demonstrate that the proposed pipeline enables
    meaningful learning despite severe data limitations, while clearly
    highlighting the fundamental challenges associated with sparse
    medical image segmentation.
    """)
elif page == "6. Conclusion & Future Work":
    st.markdown("""
    ### Conclusion and Future Work

    This project presents a complete pipeline for the analysis and segmentation
    of fetal ossification centers from volumetric CT data,
    covering all stages from raw data reconstruction to model training and evaluation.
    
    Despite the substantial effort invested in data preprocessing,
    including reverse engineering of heterogeneous medical files,
    structural analysis, and dataset construction,
    the experimental evaluation was conducted on a single model configuration.
    
    This limitation is primarily due to the complexity of the problem,
    which requires significant work not only in model design,
    but also in data engineering, preprocessing, and exploratory analysis.
    
    Unlike traditional data analysis tasks,
    medical image analysis involves spatial, geometric, and structural reasoning,
    which introduces additional layers of difficulty beyond standard statistical approaches.
    
    The results obtained in this study demonstrate that the proposed pipeline
    provides a solid foundation for addressing this challenging problem.
    The model is capable of capturing meaningful anatomical patterns,
    despite the extreme sparsity and variability of the data.
    
    However, several directions for further improvement remain.
    
    Future work will focus on:
    
    * refining ROI extraction and cropping strategies,
    * exploring different patch sizes and spatial contexts,
    * improving sampling techniques for highly imbalanced data,
    * and extending the analysis to multiple model architectures.
    
    Overall, this work should be viewed as a strong initial step
    toward solving a complex and underexplored problem in medical image segmentation,
    with significant potential for further development.
    The results confirm that the main challenge lies in data representation
    and structural sparsity, rather than model capacity alone.
    """)
    st.markdown("""
    ---
    **Author:** Michał Zieliński  
    **Project:** Fetal Ossification Segmentation  
    **Date:** 2026  
    """)