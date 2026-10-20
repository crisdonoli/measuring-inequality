import re


def clean_networth_column(column):
    """Extract numeric value from Forbes NetWorth strings.

    Args:
        column (pd.Series): Strings like '$17.7 B'.

    Returns:
        pd.Series: Float values (billions).
    """
    column = column.apply(lambda x: float(re.findall(r"[\d\.]+", x)[0]))
    return column


def convert_currency(serie, rate, to="clp"):
    """Convert net worth between USD and CLP.

    Args:
        serie (pd.Series): Values in billions (source currency).
        rate (float): CLP per USD.
        to (str): 'clp' for USD->CLP, else CLP->USD.

    Returns:
        pd.Series: Converted values.
    """
    if to == "clp":
        return serie * rate * 1e9  # USD -> CLP
    return serie / rate  # CLP -> USD
