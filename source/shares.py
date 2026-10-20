import pandas as pd
import numpy as np


def weighted_shares(df, wealth_col="net_wealth", weight_col="factor"):
    """Bottom 50%, top 10% and top 1% wealth shares (survey-weighted).

    Args:
        df (pd.DataFrame): Household data.
        wealth_col (str): Wealth column.
        weight_col (str): Survey weight column.

    Returns:
        dict: {bottom50, top10, top1} as fractions of total wealth.
    """
    wealth = df[wealth_col]
    weight = df[weight_col]
    order = wealth.argsort()
    wealth = wealth.iloc[order]
    weight = weight.iloc[order]
    total_wealth = (wealth * weight).sum()
    total_weight = weight.sum()

    cum_weight = weight.cumsum() / total_weight

    def share(mask):
        return (wealth[mask] * weight[mask]).sum() / total_wealth

    bottom50 = share(cum_weight <= 0.5)
    top10 = share(cum_weight > 0.9)
    top1 = share(cum_weight > 0.99)

    return {"bottom50": bottom50, "top10": top10, "top1": top1}


def top_share(df, wealth_col="net_wealth", weight_col="factor", p=0.99):
    """Wealth share above percentile p.

    Args:
        df (pd.DataFrame): Household data.
        wealth_col (str): Wealth column.
        weight_col (str): Weight column.
        p (float): Lower percentile threshold (e.g. 0.99 = top 1%).

    Returns:
        float: Share of total wealth held above p.
    """
    wealth = df[wealth_col]
    weight = df[weight_col]
    order = wealth.argsort()
    wealth = wealth.iloc[order]
    weight = weight.iloc[order]
    total_wealth = (wealth * weight).sum()
    total_weight = weight.sum()
    cum_weight = weight.cumsum() / total_weight
    mask = cum_weight > p
    return (wealth[mask] * weight[mask]).sum() / total_wealth


def bottom_share(df, wealth_col="net_wealth", weight_col="factor", p=0.5):
    """Wealth share at or below percentile p.

    Args:
        df (pd.DataFrame): Household data.
        wealth_col (str): Wealth column.
        weight_col (str): Weight column.
        p (float): Upper percentile threshold (e.g. 0.5 = bottom 50%).

    Returns:
        float: Share of total wealth held at/below p.
    """
    wealth = df[wealth_col]
    weight = df[weight_col]
    order = wealth.argsort()
    wealth = wealth.iloc[order]
    weight = weight.iloc[order]
    total_wealth = (wealth * weight).sum()
    total_weight = weight.sum()
    cum_weight = weight.cumsum() / total_weight
    mask = cum_weight <= p
    return (wealth[mask] * weight[mask]).sum() / total_wealth
