# =============================================================================
# explore_data.py — Puzzle Category Analysis
# Paste this into a Kaggle notebook cell to understand your training data.
# =============================================================================

import polars as pl
import re

train = pl.read_csv('/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv')
test  = pl.read_csv('/kaggle/input/nvidia-nemotron-3-reasoning-challenge/test.csv')

print(f'Train size : {len(train):,}')
print(f'Test size  : {len(test):,}')
print()

# ── Categorise puzzles by keywords in the prompt ──────────────────────────────
def categorise(prompt: str) -> str:
    p = prompt.lower()
    if 'bit' in p or 'binary' in p or 'xor' in p or 'shift' in p:
        return 'bit_manipulation'
    if 'roman' in p or 'numeral' in p:
        return 'roman_numerals'
    if 'equation' in p or 'algebra' in p or 'solve for' in p:
        return 'algebra'
    if 'encrypt' in p or 'cipher' in p or 'caesar' in p:
        return 'cipher'
    if 'sequence' in p or 'pattern' in p or 'series' in p:
        return 'sequence'
    if 'word' in p or 'anagram' in p or 'letter' in p:
        return 'word_puzzle'
    if 'grid' in p or 'matrix' in p:
        return 'grid_matrix'
    return 'other'

train = train.with_columns(
    pl.col('prompt').map_elements(categorise, return_dtype=pl.Utf8).alias('category')
)

print('=== Puzzle Category Distribution ===')
print(train.group_by('category').len().sort('len', descending=True))

# ── Answer type analysis ───────────────────────────────────────────────────────
def answer_type(answer: str) -> str:
    if re.fullmatch(r'[01]+', answer):
        return 'binary_string'
    if re.fullmatch(r'[IVXLCDM]+', answer):
        return 'roman_numeral'
    try:
        float(answer)
        return 'number'
    except ValueError:
        pass
    if len(answer.split()) > 1:
        return 'phrase'
    return 'word'

train = train.with_columns(
    pl.col('answer').map_elements(answer_type, return_dtype=pl.Utf8).alias('answer_type')
)

print()
print('=== Answer Type Distribution ===')
print(train.group_by('answer_type').len().sort('len', descending=True))

# ── Sample puzzles per category ───────────────────────────────────────────────
print()
print('=== Sample Puzzles ===')
for cat in train['category'].unique().to_list():
    sample = train.filter(pl.col('category') == cat).head(1)
    prompt_preview = sample['prompt'][0][:300]
    answer_val = sample['answer'][0]
    print(f'\n--- {cat.upper()} ---')
    print(f"Prompt: {prompt_preview}")
    print(f"Answer: {answer_val}")

print()
print('=== Prompt Length Stats ===')
train = train.with_columns(
    pl.col('prompt').str.len_chars().alias('prompt_len')
)
print(train.select(['prompt_len']).describe())
