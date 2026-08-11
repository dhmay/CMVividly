"""Utilities for constructing Hamming-1 CDR3 sequence pairs."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable

import pandas as pd

def find_hamming1_pairs_same_vj(pdf: pd.DataFrame) -> pd.DataFrame:
    # divide up pdf by [vgene, jgene] and find Hamming-1 pairs within each group
    grouped = pdf.groupby(["vgene", "jgene"])
    output_dfs = []
    for (vgene, jgene), group in grouped:
        pairs_df = find_cdr3_hamming1_pairs(group, seq_col="tcr")
        output_dfs.append(pairs_df)
    if not output_dfs:
        return pd.DataFrame(
            columns=[
                "row_i",
                "row_j",
                "tcr_i",
                "tcr_j",
                "differing_position",
                "seq_length",
            ]
        )
    return pd.concat(output_dfs, ignore_index=True)

def find_cdr3_hamming1_pairs(pdf: pd.DataFrame,
							 seq_col="cdr3") -> pd.DataFrame:
	"""Build all Hamming-1 pairs from the specified column
	in an input dataframe.

	If multiple rows have the same CDR3 sequence, all combinations of
	pairs involving those rows are included in the output.

	Args:
		pdf: Input dataframe containing the specified CDR3 column.
		seq_col: Name of the column containing CDR3 sequences.

	Returns:
		DataFrame with one row per pair of input rows whose 
		sequences are the same length and Hamming distance 1.

		Output columns:
		- ``row_i``: original index label of first row in pair
		- ``row_j``: original index label of second row in pair
		- ``{seq_col}_i``: first row's CDR3 sequence
		- ``{seq_col}_j``: second row's CDR3 sequence
		- ``differing_position``: 0-based index where the sequences differ
		- ``cdr3_length``: sequence length shared by the pair
	"""
	if seq_col not in pdf.columns:
		raise ValueError(f"Input dataframe must contain a '{seq_col}' column.")

	if pdf.empty:
		return pd.DataFrame(
			columns=[
				"row_i",
				"row_j",
				f"{seq_col}_i",
				f"{seq_col}_j",
				"differing_position",
				"cdr3_length",
			]
		)

	seq = pdf[seq_col]
	if seq.isna().any():
		raise ValueError(f"Input dataframe contains null values in '{seq_col}'.")

	seq_str = seq.astype(str)
	lengths = seq_str.str.len()

	row_index = pdf.index.to_list()
	seq_to_row_positions_by_length: dict[int, dict[str, list[int]]] = defaultdict(
		lambda: defaultdict(list)
	)

	for pos, (seq, seq_len) in enumerate(zip(seq_str.to_list(), lengths.to_list())):
		seq_to_row_positions_by_length[seq_len][seq].append(pos)

	output_rows: list[dict[str, object]] = []

	for seq_len, seq_to_positions in seq_to_row_positions_by_length.items():
		unique_sequences = list(seq_to_positions.keys())
		if len(unique_sequences) < 2:
			continue

		for seq_id_i, seq_id_j, differing_position in _iter_hamming1_unique_sequence_pairs(unique_sequences):
			seq_i = unique_sequences[seq_id_i]
			seq_j = unique_sequences[seq_id_j]
			pos_i = seq_to_positions[seq_i]
			pos_j = seq_to_positions[seq_j]

			for row_pos_i in pos_i:
				for row_pos_j in pos_j:
					output_rows.append(
						{
							"row_i": row_index[row_pos_i],
							"row_j": row_index[row_pos_j],
							f"{seq_col}_i": seq_i,
							f"{seq_col}_j": seq_j,
							"differing_position": differing_position,
							"seq_length": seq_len,
						}
					)

	if not output_rows:
		return pd.DataFrame(
			columns=[
				"row_i",
				"row_j",
				"cdr3_i",
				"cdr3_j",
				"differing_position",
				"seq_length",
			]
		)

	return pd.DataFrame(output_rows)


def populate_hamming1_pairs_othercols(pdf_all_ham1_pairs: pd.DataFrame,
									  pdf_original: pd.DataFrame) -> pd.DataFrame:
    """Populate additional columns in a dataframe of Hamming-1 pairs.

    Args:
        pdf_all_ham1_pairs: DataFrame containing Hamming-1 pairs with columns
            ``row_i`` and ``row_j`` that are index labels of the original dataframe.
        pdf_original: Original dataframe from which the pairs were derived.

    Returns:
        DataFrame with additional columns populated from the original dataframe.
    """
    pdf_original_withrow = pdf_original.copy()
    pdf_original_withrow["row"] = range(len(pdf_original_withrow))
    pdf_hamming1_pairs_allcols = (
        pdf_all_ham1_pairs
        .drop(columns=["cdr3_i", "cdr3_j"])
        .merge(pdf_original_withrow[["row", "tcr", "cdr3", "vgene", "jgene", "hla_cocluster", "hla"]], 
            left_on="row_i", right_on="row")
        .rename(columns={"tcr": "tcr_i",
                        "vgene": "vgene_i",
                        "jgene": "jgene_i",
                        "cdr3": "cdr3_i",
                        "hla_cocluster": "hla_cocluster_i",
                        "hla": "hla_i"
                        })
        .merge(pdf_original_withrow[["row", "tcr", "cdr3", "vgene", "jgene", "hla_cocluster", "hla"]], 
            left_on="row_j", right_on="row")
        .rename(columns={"tcr": "tcr_j",
                        "vgene": "vgene_j",
                        "jgene": "jgene_j",
                        "cdr3": "cdr3_j",
                        "hla_cocluster": "hla_cocluster_j",
                        "hla": "hla_j"
                        })    
    )
    pdf_hamming1_pairs_allcols["same_vgene"] = pdf_hamming1_pairs_allcols["vgene_i"] == pdf_hamming1_pairs_allcols["vgene_j"]
    pdf_hamming1_pairs_allcols["same_jgene"] = pdf_hamming1_pairs_allcols["jgene_i"] == pdf_hamming1_pairs_allcols["jgene_j"]
    pdf_hamming1_pairs_allcols["same_hlacocluster"] = pdf_hamming1_pairs_allcols["hla_cocluster_i"] == pdf_hamming1_pairs_allcols["hla_cocluster_j"]
    pdf_hamming1_pairs_allcols["same_hla"] = pdf_hamming1_pairs_allcols["hla_i"] == pdf_hamming1_pairs_allcols["hla_j"]
    return pdf_hamming1_pairs_allcols


def _iter_hamming1_unique_sequence_pairs(sequences: list[str]) -> Iterable[tuple[int, int, int]]:
	"""Yield unique sequence-id pairs that are Hamming-1 apart.

	The input sequences must all be the same length and unique.

	Yields:
		Tuples of ``(left_id, right_id, differing_position)``.
	"""
	if not sequences:
		return

	seq_len = len(sequences[0])
	masked_buckets: dict[tuple[int, str], list[int]] = defaultdict(list)

	for seq_id, seq in enumerate(sequences):
		for pos in range(seq_len):
			masked = seq[:pos] + "*" + seq[pos + 1 :]
			masked_buckets[(pos, masked)].append(seq_id)

	seen_pairs: set[tuple[int, int]] = set()
	for (pos, _masked), seq_ids in masked_buckets.items():
		if len(seq_ids) < 2:
			continue
		for left, right in combinations(seq_ids, 2):
			pair = (left, right) if left < right else (right, left)
			if pair in seen_pairs:
				continue
			seen_pairs.add(pair)
			yield pair[0], pair[1], pos


