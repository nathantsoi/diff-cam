#!/bin/bash
# Combine per-task result.tsv files without losing failed/OOM/interrupted rows.

set -euo pipefail

if (( $# < 1 || $# > 2 )); then
    echo "usage: $0 RESULT_ROOT [OUTPUT_TSV]" >&2
    exit 2
fi

result_root="$1"
output_path="${2:-$result_root/results.tsv}"

mapfile -t result_files < <(find "$result_root" -mindepth 2 -maxdepth 2 -type f -name result.tsv -print | sort)
if (( ${#result_files[@]} == 0 )); then
    echo "no per-task result.tsv files below $result_root" >&2
    exit 1
fi

tmp_path="$output_path.tmp.$$"
head -n 1 "${result_files[0]}" > "$tmp_path"
for result_file in "${result_files[@]}"; do
    tail -n +2 "$result_file"
done | sort -t $'\t' -k1,1n >> "$tmp_path"
mv "$tmp_path" "$output_path"

echo "wrote ${#result_files[@]} rows to $output_path"
awk -F '\t' 'NR == 1 { for (i=1; i<=NF; i++) col[$i]=i; next }
    { count[$(col["outcome"])]++ }
    END { for (outcome in count) printf "%s\t%d\n", outcome, count[outcome] }' \
    "$output_path" | sort
