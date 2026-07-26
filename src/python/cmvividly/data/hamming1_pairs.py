"""Utilities for constructing Hamming-1 CDR3 sequence pairs."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable

import pandas as pd

def find_cdr3_hamming1_pairs(pdf: pd.DataFrame) -> pd.DataFrame:
	"""Build all Hamming-1 CDR3 pairs from the "cdr3" column
	in an input dataframe.

	If multiple rows have the same CDR3 sequence, all combinations of
	pairs involving those rows are included in the output.

	Args:
		pdf: Input dataframe containing ``cdr3`` column.

	Returns:
		DataFrame with one row per pair of input rows whose ``cdr3``
		sequences are the same length and Hamming distance 1.

		Output columns:
		- ``row_i``: original index label of first row in pair
		- ``row_j``: original index label of second row in pair
		- ``cdr3_i``: first row's CDR3 sequence
		- ``cdr3_j``: second row's CDR3 sequence
		- ``differing_position``: 0-based index where the sequences differ
		- ``cdr3_length``: sequence length shared by the pair
	"""
	if "cdr3" not in pdf.columns:
		raise ValueError("Input dataframe must contain a 'cdr3' column.")

	if pdf.empty:
		return pd.DataFrame(
			columns=[
				"row_i",
				"row_j",
				"cdr3_i",
				"cdr3_j",
				"differing_position",
				"cdr3_length",
			]
		)

	cdr3 = pdf["cdr3"]
	if cdr3.isna().any():
		raise ValueError("Input dataframe contains null values in 'cdr3'.")

	cdr3_str = cdr3.astype(str)
	lengths = cdr3_str.str.len()

	row_index = pdf.index.to_list()
	seq_to_row_positions_by_length: dict[int, dict[str, list[int]]] = defaultdict(
		lambda: defaultdict(list)
	)

	for pos, (seq, seq_len) in enumerate(zip(cdr3_str.to_list(), lengths.to_list())):
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
							"cdr3_i": seq_i,
							"cdr3_j": seq_j,
							"differing_position": differing_position,
							"cdr3_length": seq_len,
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
				"cdr3_length",
			]
		)

	return pd.DataFrame(output_rows)


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


