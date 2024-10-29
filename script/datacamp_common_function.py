"""Commonly used functions from Datacamp"""

import random
from typing import Tuple, Dict, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sampling


def check_and_return_arguments(
    pop: pd.DataFrame,
    column_sampled: str = None,
    values_sampled: Tuple[object] = None,
    sample_size: int = None,
    sample_frac: float = None
) -> Tuple[str, list, int, int, float]:
    """Check the consitency of sample_size and sample_frac and
    return pop_size, sample_size, sample_frac.

    Args:
        pop (pd.DataFrame): Population dataset.
        column_sampled (str, optional): The column with specfic sampled values. Defaults to None.
        values_sampled (Tuple[str], optional): Rows with these values are sample. Defaults to None.
        sample_size (int, optional): Size (rows) of the desired sample. Defaults to None.
        sample_frac (float, optional): Fraction of the desired sample to the population dataset.

    Returns:
        Tuple[int, int, float]: Tuple of pop_size (int): Size (rows) of the desired population
        dataset, sample_size and sample_frac
    """
    if column_sampled is None:
        column_sampled = pop.columns[0]
    if values_sampled is None:
        values_sampled = tuple(pop[column_sampled].unique())
    pop_size, _ = pop.shape

    if sample_size is None and sample_frac is None:
        sample_size = pop_size
        sample_frac = 1
    elif sample_size is None and sample_frac is not None:
        sample_size = round(sample_frac * pop_size)
    elif sample_size is not None and sample_frac is None:
        sample_frac = sample_size / pop_size
    elif sample_size is not None and sample_frac is not None:
        assert sample_frac * pop_size == sample_size, 'Non-consistent sample_frac and sample_size'

    return column_sampled, values_sampled, pop_size, sample_size, sample_frac


def _systematic_sampling(
    pop: pd.DataFrame,
    sample_size: int = None,
    sample_frac: float = None,
    random_seed: int = 42,
    check_and_return: Callable = check_and_return_arguments
) -> pd.DataFrame:
    """Systematic_sampling.
    When population dataset is sorted, the sample dataset would be sorted
    which results in sample bias.

    Args:
        pop (pd.DataFrame): Population dataset.
        sample_size (int, optional): Size (rows) of the desired sample. Defaults to None.
        sample_frac (float, optional): Fraction of the desired sample to the population dataset.
        random_seed (int, optional): Random seed. Defaults to 42.
        check_and_return (Callable, optional): Check and return the pop_size, sample_size,
        sample_frac.
        Defaults to check_and_return_arguments.

    Returns:
        pd.DataFrame: Sample dataset.
    """
    _, _, pop_size, sample_size, _ = check_and_return(
        pop=pop,
        sample_size=sample_size,
        sample_frac=sample_frac
    )
    interval = pop_size // sample_size

    np.random.seed(random_seed)
    start = random.randint(0, interval)

    return pop[start::interval]


def safe_systematic_sampling(
    pop: pd.DataFrame,
    replace: bool = False,
    sample_size: int = None,
    sample_frac: float = None,
    random_seed: int = 42,
    sampling: Callable = _systematic_sampling
) -> pd.DataFrame:
    """Systematic_sampling.
    When population dataset is sorted, the sample dataset would be not sorted
    and still shuffled which results in no sample bias and is equivelent to simple sampling.

    Args:
        pop (pd.DataFrame): Population dataset.
        replace (bool, optional): Whether sample with replacement. Defaults to False.
        sample_size (int, optional): Size (rows) of the desired sample. Defaults to None.
        sample_frac (float, optional): Fraction of the desired sample to the population
        dataset. Defaults to None.
        random_seed (int, optional): Random seed. Defaults to 42.
        sampling (Callable, optional): Base sampling. Default to _systematic_sampling.

    Returns:
        pd.DataFrame: Sample dataset.
    """
    # Shuffle the rows of pop
    pop_shuffled = pop.sample(frac=1, replace=replace)
    # Reset the row indexes and create an index column
    pop_shuffled = pop_shuffled.reset_index(drop=True).reset_index()

    return sampling(
        pop=pop_shuffled,
        sample_size=sample_size,
        sample_frac=sample_frac,
        random_seed=random_seed,
        )


def simple_sampling(
    pop: pd.DataFrame,
    column_sampled: str = None,
    values_sampled: Tuple[object] = None,
    replace: bool = False,
    sample_size: int = None,
    sample_frac: float = None,
    random_seed: int = 42,
    check_and_return: Callable = check_and_return_arguments
) -> pd.DataFrame:
    """Simple sampling to sample rows with specific values in a columns.

    Args:
        pop (pd.DataFrame): Population dataset.
        column_sampled (str, optional): The column with specfic sampled values. Defaults to None.
        values_sampled (Tuple[str], optional): Rows with these values are sample. Defaults to None.
        replace (bool, optional): Whether sample with replacement. Defaults to False.
        sample_size (int, optional): Size (rows) of the desired sample. Defaults to None.
        sample_frac (float, optional): Fraction of the desired sample to the population
        dataset. Defaults to None.
        random_seed (int, optional): Random seed. Defaults to 42.
        check_and_return (Callable, optional): Check and return the pop_size, sample_size, sample_frac.
        Defaults to check_and_return_arguments.

    Returns:
        pd.DataFrame: Sample dataset.
    """
    column_sampled, values_sampled, _, _, sample_frac = check_and_return(
        pop=pop,
        column_sampled=column_sampled,
        values_sampled=values_sampled,
        sample_size=sample_size,
        sample_frac=sample_frac
    )

    rows_sampled = pop[column_sampled].isin(values_sampled)
    pop_sampled = pop[rows_sampled]

    return pop_sampled.sample(frac=sample_frac, replace=replace, random_state=random_seed)


def stratified_sampling(
    pop: pd.DataFrame,
    column_sampled: str = None,
    values_sampled: Tuple[object] = None,
    replace: bool = False,
    sample_size: int = None,
    sample_frac: float = None,
    random_seed: int = 42,
    check_and_return: Callable = check_and_return_arguments
) -> pd.DataFrame:
    """Stratified sampling to sample rows with specific values in a columns.
    Split the population into subgroups and do simple sampling on each subgroup.

    Args:
        pop (pd.DataFrame): Population dataset.
        column_sampled (str, optional): The column with specfic sampled values. Defaults to None.
        values_sampled (Tuple[str], optional): Rows with these values are sample. Defaults to None.
        replace (bool, optional): Whether sample with replacement. Defaults to False.
        sample_size (int, optional): Size (rows) of the desired sample. Defaults to None.
        sample_frac (float, optional): Fraction of the desired sample to the population dataset.
        Defaults to None.
        random_seed (int, optional): Random seed. Defaults to 42.
        check_and_return (Callable, optional): Check and return the pop_size, sample_size, sample_frac.
        Defaults to check_and_return_arguments.

    Returns:
        pd.DataFrame: Sample dataset.
    """
    column_sampled, values_sampled, _, _, sample_frac = check_and_return(
        pop=pop,
        column_sampled=column_sampled,
        values_sampled=values_sampled,
        sample_size=sample_size,
        sample_frac=sample_frac
    )

    rows_sampled = pop[column_sampled].isin(values_sampled)
    pop_sampled = pop[rows_sampled]
    pop_sampled[column_sampled] = pop_sampled[column_sampled].cat.remove_unused_categories()
    return pop_sampled.groupby(column_sampled) \
        .sample(frac=sample_frac, replace=replace, random_state=random_seed)


def equally_stratified_sampling(
    pop: pd.DataFrame,
    column_sampled: str = None,
    values_sampled: Tuple[object] = None,
    replace: bool = False,
    sample_size: int = None,
    sample_frac: float = None,
    random_seed: int = 42,
    check_and_return: Callable = check_and_return_arguments
) -> pd.DataFrame:
    """Equally stratified sampling with equal samples on rows with different values.

    Args:
        pop (pd.DataFrame): Population dataset.
        column_sampled (str, optional): The column with specfic sampled values. Defaults to None.
        values_sampled (Tuple[str], optional): Rows with these values are sample. Defaults to None.
        replace (bool, optional): Whether sample with replacement. Defaults to False.
        sample_size (int, optional): Size (rows) of the desired sample. Defaults to None.
        sample_frac (float, optional): Fraction of the desired sample to the population dataset.
        Defaults to None.
        random_seed (int, optional): Random seed. Defaults to 42.
        check_and_return (Callable, optional): Check and return the pop_size, sample_size, sample_frac.
        Defaults to check_and_return_arguments.

    Returns:
        pd.DataFrame: Sample dataset.
    """
    column_sampled, values_sampled, _, sample_size, _ = check_and_return(
        pop=pop,
        column_sampled=column_sampled,
        values_sampled=values_sampled,
        sample_size=sample_size,
        sample_frac=sample_frac
    )

    rows_sampled = pop[column_sampled].isin(values_sampled)
    pop_sampled = pop[rows_sampled]
    pop_sampled[column_sampled] = pop_sampled[column_sampled].cat.remove_unused_categories()
    return pop_sampled.groupby(column_sampled) \
        .sample(n=sample_size, replace=replace, random_state=random_seed)


def weightedly_stratified_sampling(
    pop: pd.DataFrame,
    column_sampled: str = None,
    values_sampled: Tuple[object] = None,
    weights: Dict[str, float] = None,
    replace: bool = False,
    sample_size: int = None,
    sample_frac: float = None,
    random_seed: int = 42,
    check_and_return: Callable = check_and_return_arguments
) -> pd.DataFrame:
    """Weightedly stratified sampling with different weights (proportion)
    on rows with different values.

    Args:
        pop (pd.DataFrame): Population dataset.
        column_sampled (str, optional): The column with specfic sampled values. Defaults to None.
        values_sampled (Tuple[str], optional): Rows with these values are sample. Defaults to None.
        weights (Dict[str, float], optional): The dict of value_sampled-value_weight pairs.
        value_weight is ralative to average_weight.
        replace (bool, optional): Whether sample with replacement. Defaults to False.
        sample_size (int, optional): Size (rows) of the desired sample. Defaults to None.
        sample_frac (float, optional): Fraction of the desired sample to the population dataset.
        Defaults to None.
        random_seed (int, optional): Random seed. Defaults to 42.
        check_and_return (Callable, optional): Check and return the pop_size, sample_size, sample_frac.
        Defaults to check_and_return_arguments.

    Returns:
        pd.DataFrame: Sample dataset.
    """
    column_sampled, values_sampled, _, _, sample_frac = check_and_return(
        pop=pop,
        column_sampled=column_sampled,
        values_sampled=values_sampled,
        sample_size=sample_size,
        sample_frac=sample_frac
    )

    if weights is None:
        weights = {value_sampled: 1 / len(values_sampled) for value_sampled in values_sampled}

    for value_sampled in values_sampled:
        row_sampled = pop[column_sampled] == value_sampled
        value_weight = weights[value_sampled]
        average_weight = np.mean(tuple(weights.values()))
        pop['weight'] = np.where(row_sampled, value_weight, average_weight)

    return pop.sample(frac=sample_frac, replace=replace, weights='weight', random_state=random_seed)


def cluster_sampling(
    pop: pd.DataFrame,
    column_sampled: str = None,
    replaces: Tuple[bool, bool] = None,
    sample_size: int = None,
    sample_frac: float = None,
    subgroup_size: int = None,
    random_seed: int = 42,
    sampling: Callable = equally_stratified_sampling
) -> pd.DataFrame:
    """Cluster sampling with different weights (proportion) on rows with different values.
    Do simple sampling to pick subgroups and do simple sampling only on these subgroups.
    A special case of multistage sampling.

    Args:
        pop (pd.DataFrame): Population dataset.
        column_sampled (str, optional): The column with specfic sampled values. Default to None.
        replaces (Tuple[bool, bool], optional): Whether subgroup and sample with replacement.
        Defaults to (False, False).
        sample_size (int, optional): Size (rows) of the desired sample. Defaults to None.
        sample_frac (float, optional): Fraction of the desired sample to the population dataset.
        Defaults to None.
        subgroup_size (int, optional): Size (count) of subgroups. Default to None.
        random_seed (int, optional): Random seed. Defaults to 42.
        sampling (Callable, optional): Base sampling. Default to equally_stratified_sampling.

    Returns:
        pd.DataFrame: Sample dataset.
    """
    if replaces is None:
        replaces = (False, False)

    values = tuple(pop[column_sampled].unique())

    if subgroup_size is None:
        subgroup_size = len(values)

    if replaces[0]:
        values_sampled = random.choices(values, k=subgroup_size)
    else:
        values_sampled = random.sample(values, k=subgroup_size)

    return sampling(
        pop=pop,
        column_sampled=column_sampled,
        values_sampled=values_sampled,
        replace=replaces[1],
        sample_size=sample_size,
        sample_frac=sample_frac,
        random_seed=random_seed,
    )


def mutistage_sampling(
    pop: pd.DataFrame,
    columns_sampled_and_size: Dict[str, int] = None,
    replaces: Tuple[bool, bool] = None,
    sample_size: int = None,
    sample_frac: float = None,
    random_seed: int = 42,
    monostage_sampling: Callable = cluster_sampling
) -> pd.DataFrame:
    """Cluster sampling with different weights (proportion) on rows with different values.
    Do simple sampling to pick subgroups and do simple sampling only on these subgroups.
    A special case of multistage sampling.

    Args:
        pop (pd.DataFrame): Population dataset.
        columns_sampled_and_size (Dict[str, int], optional): The dict of column-which has specfic sampled
        values-size-which is the subgroup (sampled values) size of each column-pair.
        replaces (Tuple[bool, bool], optional): Whether subgroup and sample with replacement.
        Defaults to (False, False).
        sample_size (int, optional): Size (rows) of the desired sample. Defaults to None.
        sample_frac (float, optional): Fraction of the desired sample to the population dataset.
        Defaults to None.
        random_seed (int, optional): Random seed. Defaults to 42.
        monostage_sampling (Callable, optional): Base sampling for each stage. Defaults to cluster_sampling.

    Returns:
        pd.DataFrame: Sample dataset.
    """
    if replaces is None:
        replaces = (False, False)

    if columns_sampled_and_size is None:
        columns_sampled_and_size = {
            column_sampled: len(tuple(pop[column_sampled].unique()))
            for column_sampled in pop.columns
        }

    for column_sampled, subgroup_size in columns_sampled_and_size.items():
        sample = monostage_sampling(
            pop=pop,
            column_sampled=column_sampled,
            replaces=replaces,
            sample_size=sample_size,
            sample_frac=sample_frac,
            subgroup_size=subgroup_size,
            random_seed=random_seed
        )

    return sample


def samples_distributing(
    pop: pd.DataFrame,
    replace: bool = False,
    sample_size: int = None,
    sample_frac: float = None,
    samples_count: int = 500,
    random_seeds: Tuple[int] = None,
    sampling: Callable = simple_sampling,
    stats: Tuple[object] = None,
    plot: bool = False,
    **kwargs: Dict[str, object]
) -> pd.DataFrame:
    """Generates PMF for specified statistics (sample distribution).

    Args:
        pop (pd.DataFrame): Population dataset.
        replace (bool, optional): Whether sample with replacement. Defaults to False.
        sample_size (int, optional): Size (rows) of the desired sample. Defaults to None.
        sample_frac (float, optional): Fraction of the desired sample to the population dataset.
        samples_count (int, optional): The quantity of samples in sample distribution. Defaults to 500.
        random_seeds (Tuple[int], optional): List of random seeds.
        Length of it should equal to samples_count. Defaults to None.
        sampling (Callable, optional): Different sampling functions.
        Defaults to simple_sampling.
        Select in [
            systematic_sampling,
            safe_systematic_sampling,
            simple_sampling,
            stratified_sampling,
            equally_stratified_sampling,
            weightedly_stratified_sampling,
            cluster_sampling,
            mutistage_sampling
            ].
        stats (Tuple[str], optional): Statistics applied to each columns of pop. Defaults to None.
        Select in ['max', 'min', 'sum', 'mod', 'median', 'mean', 'var', 'std',
        lambda samples: np.sqrt(sample_size) * samples.var(ddof=1), # Estimated population stat
        lambda samples: np.sqrt(sample_size) * samples.std(ddof=1)]. # Estimated population stat
        plot (bool, optional): Whether plot the resample distribution for each statistic of each column.

    Returns:
        DataFrame containing statistics for each column in pop.
    """
    if random_seeds is None:
        random_seeds = tuple(range(samples_count))
    else:
        assert len(random_seeds) == samples_count, \
            'Nonconsistent length of random seeds list and samples count.'

    if stats is None:
        stats = ['mean', 'var']

    samples_stats = {col: pd.DataFrame(columns=stats) for col in pop.columns}

    for random_seed in random_seeds:
        sample = sampling(
            pop=pop,
            replace=replace,
            sample_size=sample_size,
            sample_frac=sample_frac,
            random_seed=random_seed,
            **kwargs
        )
        sample_stats = sample.agg(stats)
        for col in pop.columns:
            sample_stats_col = sample_stats[[col]].T.reset_index(drop=True)
            samples_stats[col] = pd.concat([samples_stats[col], sample_stats_col])

    if plot:
        num_stats = len(stats)
        num_cols = len(pop.columns)
        fig, axes = plt.subplots(
            nrows=num_stats,
            ncols=num_cols,
            figsize=(5 * num_cols, 5 * num_stats),
            squeeze=False
        )
        fig.suptitle('Statistics plots for each column')

        for i, stat in enumerate(stats):
            for j, col in enumerate(pop.columns):
                axes[i, j].hist(samples_stats[col][stat], bins=samples_count // 10, density=True)
                axes[i, j].set_title(f'Distribution of {stat} for {col}')
                axes[i, j].set_xlabel(stat)
                axes[i, j].set_ylabel('Relative Frequency')
                axes[i, j].grid(axis='y')

        plt.tight_layout()
        plt.show()

    return samples_stats


def bootstrapping(
    sample: pd.DataFrame,
    replace: bool = True,
    sample_size: int = None,
    sample_frac: float = None,
    samples_count: int = 500,
    random_seeds: Tuple[int] = None,
    stats: Tuple[object] = None,
    plot: bool = False,
    distributing: Callable = samples_distributing,
    **kwargs: Dict[str, object]
) -> pd.DataFrame:
    """Generates PMF for specified statistics (resample distribution,
    i.e. resample from samples with replacement).

    Args:
        sample (pd.DataFrame): Sample dataset.
        replace (bool, optional): Whether sample with replacement. Defaults to False.
        sample_size (int, optional): Size (rows) of the desired sample. Defaults to None.
        sample_frac (float, optional): Fraction of the desired sample to the population dataset.
        samples_count (int, optional): The quantity of samples in sample distribution. Defaults to 500,
        random_seeds (Tuple[int], optional): List of random seeds.
        Length of it should equal to samples_count. Defaults to None.
        stats (Tuple[str], optional): Statistics applied to each columns of pop. Defaults to None.
        Select in ['max', 'min', 'sum', 'mean', 'var', 'std',
        lambda samples: np.sqrt(sample_size) * samples.var(ddof=1), # Estimated population stat
        lambda samples: np.sqrt(sample_size) * samples.std(ddof=1)]. # Estimated population stat
        plot (bool, optional): Whether plot the resample distribution for each statistic of each column.
        distributing (Callable, optional): Based samples distributing.
        Defaults to samples_distributing.

    Returns:
        pd.DataFrame: Resampled DataFrame containing statistics for each column in sample.
    """
    return distributing(
        pop=sample,
        replace=replace,
        sample_size=sample_size,
        sample_frac=sample_frac,
        samples_count=samples_count,
        random_seeds=random_seeds,
        stats=stats,
        plot=plot,
        **kwargs
    )


def confidence_interval(
    pop: pd.DataFrame,
    replace: bool = False,
    sample_size: int = None,
    sample_frac: float = None,
    samples_count: int = 500,
    random_seeds: Tuple[int] = None,
    stats: Tuple[object] = None,
    distributing: Callable = samples_distributing,
    alpha_interval: Tuple[float] = None,
    **kwargs: Dict[str, object]
) -> np.ndarray:
    """Generates PMF for specified statistics (sample distribution).

    Args:
        pop (pd.DataFrame): Population dataset.
        replace (bool, optional): Whether sample with replacement. Defaults to False.
        sample_size (int, optional): Size (rows) of the desired sample. Defaults to None.
        sample_frac (float, optional): Fraction of the desired sample to the population dataset.
        samples_count (int, optional): The quantity of samples in sample distribution. Defaults to 500.
        random_seeds (Tuple[int], optional): List of random seeds.
        Length of it should equal to samples_count. Defaults to None.
        stats (Tuple[str], optional): _description_. Defaults to None.
        Select in ['max', 'min', 'sum', 'mean', 'var', 'std',
        lambda samples: np.sqrt(sample_size) * samples.var(ddof=1), # Estimated population stat
        lambda samples: np.sqrt(sample_size) * samples.std(ddof=1)]. # Estimated population stat
        distributing (Callable, optional): Based samples distributing.
        Defaults to samples_distributing.

    Returns:
        ndarray as a confidence interval.
    """
    if alpha_interval is None:
        alpha_interval = [0.05, 0.95]

    samples = distributing(
        pop=pop,
        replace=replace,
        sample_size=sample_size,
        sample_frac=sample_frac,
        samples_count=samples_count,
        random_seeds=random_seeds,
        stats=stats,
        **kwargs
    )

    return np.quantile(samples, alpha_interval)
