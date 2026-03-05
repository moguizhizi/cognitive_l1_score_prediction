# src/data/splitter.py

from sklearn.model_selection import train_test_split


def split_dataset(
    df,
    test_size=0.2,
    val_size=0.1,
    random_state=42,
):
    """
    划分 train / val / test
    """

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state,
    )

    return train_df, val_df, test_df
