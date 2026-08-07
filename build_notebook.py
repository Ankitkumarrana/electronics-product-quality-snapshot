from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "electronics_product_analysis.ipynb"


def markdown(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

nb.cells = [
    markdown(
        """# Electronics Product Quality Snapshot

This notebook prepares a small, reviewable product quality dataset for the Transformational AI machine learning team. The workflow uses a 100 row stream from Amazon Electronics reviews and a 100 row stream from product metadata. The result focuses on products with a title, an average rating of at least 4.5, and a usable price.

The analysis keeps the data pull intentionally small so the workflow is fast to inspect while still demonstrating a safe pattern for much larger files. The final table is saved in both CSV and Parquet formats for downstream analysis."""
    ),
    markdown(
        """## 1. Environment and library check

The required libraries are imported before analysis begins. Version information is printed so the execution environment can be checked quickly. If a fresh environment is being used, the following command installs the project dependencies:

```python
%pip install datasets pandas matplotlib seaborn pyarrow
```"""
    ),
    code(
        """import importlib.metadata as importlib_metadata
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset


required_packages = {
    "datasets": "datasets",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "pyarrow": "pyarrow",
}

os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"
print("Command line version checks:")
!python --version
!jupyter-notebook --version
print(f"Python: {sys.version.split()[0]}")
print(f"Python executable: {sys.executable}")
try:
    notebook_version = importlib_metadata.version("notebook")
except importlib_metadata.PackageNotFoundError:
    notebook_version = "not installed"
print(f"Jupyter Notebook package: {notebook_version}")

missing_packages = []
for package_name, distribution_name in required_packages.items():
    try:
        version = importlib_metadata.version(distribution_name)
        print(f"{package_name}: {version}")
    except importlib_metadata.PackageNotFoundError:
        missing_packages.append(package_name)

assert not missing_packages, f"Missing required packages: {missing_packages}"
assert sys.version_info >= (3, 9), "Python 3.9 or newer is required."
print("Environment check passed.")"""
    ),
    markdown(
        """## 2. Review sample

The source is streamed from the `raw_review_Electronics` configuration. Only the first 100 records are collected. `streaming=True` and `trust_remote_code=True` are explicit boolean settings, and the loop stops as soon as the sample limit is reached."""
    ),
    code(
        """sample_limit = 100
stream_reviews = True
allow_dataset_code = True

reviews_dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Electronics",
    split="full",
    streaming=stream_reviews,
    trust_remote_code=allow_dataset_code,
)

review_rows = []
for index, review in enumerate(reviews_dataset, start=1):
    review_rows.append(review)
    if index == sample_limit:
        break

reviews_df = pd.DataFrame(review_rows)
print(f"Review rows collected: {len(reviews_df)}")
reviews_df.head()"""
    ),
    code(
        """Items_To_Print = 10
pd.set_option("display.max_rows", Items_To_Print)
pd.set_option("display.max_columns", Items_To_Print)
pd.set_option("display.max_colwidth", 100)
print(f"Showing the first {Items_To_Print} review items:")
reviews_df.head(Items_To_Print)"""
    ),
    markdown(
        """## 3. Product metadata sample

The product stream uses the same 100 row limit. Product metadata contains the title, average rating, and price fields needed for the final dataset."""
    ),
    code(
        """metadata_limit = 100
stream_metadata = True
allow_metadata_code = True

metadata_dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Electronics",
    split="full",
    streaming=stream_metadata,
    trust_remote_code=allow_metadata_code,
)

metadata_rows = []
for index, item in enumerate(metadata_dataset, start=1):
    metadata_rows.append(item)
    if index == metadata_limit:
        break

item_metadata_df = pd.DataFrame(metadata_rows)
print(f"Metadata rows collected: {len(item_metadata_df)}")
item_metadata_df.head()"""
    ),
    code(
        """print("Reviews sample columns and data types:")
print(reviews_df.head().dtypes)
print("\\nMetadata sample columns and data types:")
print(item_metadata_df.head().dtypes)"""
    ),
    markdown(
        """### Safe product lookup

Prices can be missing or stored as text. The lookup function uses `try` and `except` so a missing product, missing title, or invalid price returns a readable message instead of stopping the notebook."""
    ),
code(
        """def get_product(parent_asin, metadata_rows):
    '''Return a product title and numeric price, or a useful error message.'''
    try:
        matches = [item for item in metadata_rows if item.get("parent_asin") == parent_asin]
        if not matches:
            raise KeyError(f"No product found for parent_asin={parent_asin}")

        product = matches[0]
        title = product.get("title")
        if title is None or not str(title).strip():
            raise ValueError("The product title is missing")

        numeric_price = pd.to_numeric(product.get("price"), errors="coerce")
        if pd.isna(numeric_price):
            raise ValueError("The product price is missing or not numeric")

        return {"title": str(title).strip(), "price": float(numeric_price)}
    except (KeyError, TypeError, ValueError) as error:
        return f"Product lookup failed: {error}"


example_parent_asin = item_metadata_df.iloc[0]["parent_asin"]
print("Valid lookup:", get_product(example_parent_asin, metadata_rows))
print("Invalid lookup:", get_product("missing_parent_asin", metadata_rows))

numeric_prices = pd.to_numeric(item_metadata_df["price"], errors="coerce")
print(f"Metadata items with a usable price: {numeric_prices.notna().sum()}")
print(f"Metadata items without a usable price: {numeric_prices.isna().sum()}")"""
    ),
    markdown("""## 4. Compare and summarize the data objects"""),
    code(
        """dataframes_to_review = {
    "reviews_df": reviews_df,
    "item_metadata_df": item_metadata_df,
}

for dataframe_name, dataframe in dataframes_to_review.items():
    print(f"{dataframe_name} columns:")
    for column_name in dataframe.columns:
        print(column_name)
    print()

print("Review data types:")
print(reviews_df.dtypes)
print("\\nMetadata data types:")
print(item_metadata_df.dtypes)"""
    ),
    code(
        """print("Reviews DataFrame summary:")
reviews_df.info()

print("\\nMetadata DataFrame summary:")
item_metadata_df.info()"""
    ),
    markdown(
        """## 5. Clean the product table

The three required fields are retained as `title`, `average_rating`, and `price`. The price column is converted with `errors="coerce"`, which changes missing or not numeric values to `NaN`. Those rows can then be excluded safely with a boolean filter."""
    ),
    code(
        """item_metadata_df["price"] = pd.to_numeric(item_metadata_df["price"], errors="coerce")
average_rating_filter = 4.5

has_title = item_metadata_df["title"].notna() & item_metadata_df["title"].astype(str).str.strip().ne("")
has_required_rating = item_metadata_df["average_rating"].ge(average_rating_filter)
has_price = item_metadata_df["price"].notna()

clean_product_mask = has_title & has_required_rating & has_price
item_metadata_cleaned_df = item_metadata_df.loc[
    clean_product_mask, ["title", "average_rating", "price"]
].copy()
top_products_df = item_metadata_cleaned_df.reset_index(drop=True)

print(f"Clean products: {len(top_products_df)} of {len(item_metadata_df)}")
top_products_df.head(10)"""
    ),
code(
        """def calculate_percentage(part, whole):
    '''Return part as a percentage of whole, with a safe zero check.'''
    if whole == 0:
        return 0.0
    return (part / whole) * 100


clean_percentage = calculate_percentage(len(top_products_df), len(item_metadata_df))
print(f"Percentage of metadata rows retained: {clean_percentage:.2f}%")"""
    ),
    markdown(
        """### Export the analysis ready files

Both formats contain the same three cleaned columns. CSV is convenient for quick inspection, while Parquet keeps typed columns and is efficient for larger analytical workflows."""
    ),
    code(
        """csv_path = Path("top_products.csv")
parquet_path = Path("top_products.parquet")

top_products_df.to_csv(csv_path, index=False)
top_products_df.to_parquet(parquet_path, index=False)

print(f"Saved {csv_path}: {csv_path.stat().st_size:,} bytes")
print(f"Saved {parquet_path}: {parquet_path.stat().st_size:,} bytes")
print("CSV columns:", pd.read_csv(csv_path).columns.tolist())
print("Parquet columns:", pd.read_parquet(parquet_path).columns.tolist())"""
    ),
    code(
        """top_rated_titles = []
for title in top_products_df["title"]:
    top_rated_titles.append(title)

print("Top rated product titles:")
for title in top_rated_titles:
    print(title)"""
    ),
    markdown("""## 6. Visual findings"""),
    code(
        """plt.figure(figsize=(10, 5))
sns.histplot(data=top_products_df, x="price", kde=True, color="#2563eb")
plt.title("Price distribution for highly rated products")
plt.xlabel("Price")
plt.ylabel("Number of products")
plt.tight_layout()
plt.show()"""
    ),
    code(
        """plt.figure(figsize=(10, 5))
sns.scatterplot(
    data=top_products_df,
    x="price",
    y="average_rating",
    hue="average_rating",
    palette="viridis",
    legend=False,
    s=90,
)
plt.title("Price and average rating")
plt.xlabel("Price")
plt.ylabel("Average rating")
plt.tight_layout()
plt.show()"""
    ),
    code(
        """figure, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=top_products_df, x="average_rating", discrete=True, ax=axes[0], color="#0f766e")
axes[0].set_title("Rating distribution")
axes[0].set_xlabel("Average rating")
axes[0].set_ylabel("Number of products")

sns.scatterplot(data=top_products_df, x="price", y="average_rating", ax=axes[1], color="#ea580c", s=80)
axes[1].set_title("Price compared with rating")
axes[1].set_xlabel("Price")
axes[1].set_ylabel("Average rating")

figure.suptitle("Electronics quality dashboard", y=1.03, fontsize=15)
figure.tight_layout()
plt.show()"""
    ),
    markdown(
        """### Focused view: ratings of 4.7 and above

The cutoff narrows the table to a small set of products that may be useful for a premium recommendation or satisfaction focused review."""
    ),
    code(
        """highly_rated_products_df_cutoff = 4.7
highly_rated_products_df = (
    top_products_df.loc[top_products_df["average_rating"].ge(highly_rated_products_df_cutoff)]
    .sort_values(["average_rating", "price"], ascending=[False, True])
    .reset_index(drop=True)
)

print(f"Products at or above {highly_rated_products_df_cutoff}: {len(highly_rated_products_df)}")
highly_rated_products_df"""
    ),
    code(
        """plt.figure(figsize=(10, 5))
sns.scatterplot(
    data=highly_rated_products_df,
    x="price",
    y="average_rating",
    size="price",
    hue="average_rating",
    palette="magma",
    sizes=(80, 260),
    legend=False,
)
plt.title(f"Products with average rating ≥ {highly_rated_products_df_cutoff}")
plt.xlabel("Price")
plt.ylabel("Average rating")
plt.tight_layout()
plt.show()"""
    ),
    markdown("""### A small business focused summary"""),
    code(
        """price_summary = (
    top_products_df.assign(
        price_band=pd.cut(
            top_products_df["price"],
            bins=[-float("inf"), 25, 100, float("inf")],
            labels=["Under $25", "$25–$100", "Over $100"],
        )
    )
    .groupby("price_band", observed=False)
    .agg(product_count=("title", "size"), average_rating=("average_rating", "mean"))
    .reset_index()
)
price_summary"""
    ),
    markdown(
        """## 7. Reflection

1. Which transformation made the dataset safer to use?
2. What is one useful pattern to investigate further?
3. How can the cleaned table support future machine learning work?
4. Why was a 100 row sample limit used for both datasets?
5. Why are both CSV and Parquet formats useful?
6. What did the visualizations reveal about prices and ratings?
7. How can this project be presented in a portfolio?"""
    ),
    code(
        """reflection_answers = {
    "1. Which transformation made the dataset safer to use?": (
        "Converting price with pd.to_numeric(errors='coerce') made the filter safer. "
        "Text such as 'None' became a missing value that could be removed with a clear boolean condition."
    ),
    "2. What is one useful pattern to investigate further?": (
        "The next useful check is whether price bands show different rating patterns. "
        "The current sample describes highly rated products, but a larger sample is needed before making a pricing claim."
    ),
    "3. How can the cleaned table support future machine learning work?": (
        "The table provides consistent product titles, numeric prices, and numeric ratings. "
        "Those fields can be joined with review text and product identifiers later for recommendation, "
        "ranking, or customer satisfaction models."
    ),
    "4. Why was a 100 row sample limit used for both datasets?": (
        "The source files are very large, so a 100 row limit keeps exploration fast and avoids loading more data than the early analysis needs. "
        "The same limit also makes the review and metadata checks easy to repeat."
    ),
    "5. Why are both CSV and Parquet formats useful?": (
        "CSV is simple to open and share across many tools. Parquet preserves useful column types and is more efficient for analytical workflows, "
        "so keeping both formats supports quick inspection and future processing."
    ),
    "6. What did the visualizations reveal about prices and ratings?": (
        "The cleaned sample contains products across several price levels while all products meet the high rating filter. "
        "The charts help compare the price spread with the rating concentration, but the small sample is not enough to claim that price causes higher ratings."
    ),
    "7. How can this project be presented in a portfolio?": (
        "The project can be presented as a compact data preparation case study. "
        "It demonstrates bounded streaming, pandas cleaning, exception handling, export design, and visual communication in one repeatable workflow."
    ),
}

reflection_lines = ["Electronics Product Quality Snapshot — Reflection\\n"]
for question, answer in reflection_answers.items():
    reflection_lines.append(f"{question}\\nAnswer: {answer}\\n")

Path("user_inputs.txt").write_text("\\n".join(reflection_lines), encoding="utf-8")
print(Path("user_inputs.txt").read_text(encoding="utf-8"))"""
    ),
    markdown(
        """## Conclusion

The workflow produces a small, typed product table that is easy to inspect and ready for a larger data pull. The exported files, visual checks, safe exception handling, and reflection notes provide a clear handoff for future modeling work."""
    ),
]

nbf.write(nb, NOTEBOOK_PATH)
print(f"Created {NOTEBOOK_PATH}")
