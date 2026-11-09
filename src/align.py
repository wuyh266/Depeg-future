def align_hour(df, col):

    df[col] = df[col].astype(int)

    df["hour"] = df[col] // 3600 * 3600

    return df
