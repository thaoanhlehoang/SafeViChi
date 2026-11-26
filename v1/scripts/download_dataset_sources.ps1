$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# ViHSD is gated: sign in with `hf auth login` and accept the dataset's access
# conditions on Hugging Face before running this script.
hf download uitnlp/vihsd `
    --type dataset `
    --revision 88e81b36ca376867640dad9df295e0f7d499ab80 `
    --include train.csv `
    --local-dir data/raw/vihsd

# Keep the VOZ-HSD usage notice next to the local build. The 10.7M records are
# streamed from pinned converted Parquet by src.dataset_builder.build.
hf download tarudesu/VOZ-HSD `
    --type dataset `
    --revision 923915f5f633c503babf2c6dcf7003593318b04f `
    --include README.md `
    --local-dir data/raw/voz_hsd_metadata
