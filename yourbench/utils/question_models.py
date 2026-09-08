"""Lossless question dataset assembly."""


def question_dataset(rows: list[dict]):
    """Build a lossless Arrow table or report incompatible schema columns safely."""
    import pyarrow as pa

    from datasets import Dataset

    columns = dict.fromkeys(key for row in rows for key in row)
    aligned = [{key: row.get(key) for key in columns} for row in rows]
    try:
        return Dataset.from_list(aligned)
    except (pa.ArrowException, TypeError):
        conflicts = []
        for key in columns:
            try:
                pa.array([row.get(key) for row in rows])
            except (pa.ArrowException, TypeError):
                conflicts.append(key)
        names = ", ".join(conflicts) or "unknown column"
        # Arrow's exception includes example values. Do not expose raw payloads.
        raise ValueError(
            f"Question schema conflict in columns: {names}. "
            "Use compatible field types across question schemas or export them separately."
        ) from None
