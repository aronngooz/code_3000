import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    merged = pd.merge(
        anon_df,
        aux_df,
        on=["age", "gender", "zip3"],
        how="inner"
    )
    counts = merged.groupby("anon_id").size()
    ids = counts[counts == 1].index

    matches = merged[merged["anon_id"].isin(ids)]

    result = matches[["anon_id", "name"]]
    result = result.rename(columns={"name": "matched_name"})

    return result


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return len(matches_df) / len(anon_df)
