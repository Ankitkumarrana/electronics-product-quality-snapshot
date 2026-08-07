# Electronics Product Quality Snapshot

This project streams a small sample of Amazon Electronics reviews and product metadata, then prepares a clean table of highly rated products with usable prices.

## Deliverables

1. `electronics_product_analysis.ipynb` contains the executed analysis notebook
2. `user_inputs.txt` contains reflection answers written by the notebook
3. `top_products.csv` contains cleaned product data in CSV format
4. `top_products.parquet` contains the same cleaned data in Parquet format

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook electronics_product_analysis.ipynb
```

The notebook uses streaming and stops after 100 rows from each source. A network connection is required for the first dataset load.
