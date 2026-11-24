import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold)
os.chdir('Path_to_directory')
os.getcwd()
from sklearn.preprocessing import PowerTransformer, MinMaxScaler


expression_matrix = pd.read_csv(
    'GSE74793_expression_matrix.csv',
    index_col=0
)

data = expression_matrix [
    expression_matrix ['label'].isin(['Control', 'D/or/H'])
].copy()

gss = GroupShuffleSplit(
    n_splits=1000,
    test_size=0.30,
    random_state=42
)

overall_class_prop = y.value_counts(normalize=True).sort_index()

candidate_splits = []

series_matrix = pd.read_csv(
    'series_matrix.csv'
)

series_matrix = series_matrix.set_index('sample_ID')

metadata = series_matrix.loc[data.index].copy()

assert all(metadata.index == data.index)

print("\nMetadata successfully matched.")

X = data.drop(columns='label').copy()

y = data['label'].copy()

groups = metadata['time'].copy()

print("\nNumber of samples:", X.shape[0])
print("Number of genes:", X.shape[1])
print("Number of time groups:", groups.nunique())

print("\nTime groups:")
print(sorted(groups.unique()))

for train_idx, test_idx in gss.split(
    X,
    y,
    groups=groups
):

    test_class_prop = (
        y.iloc[test_idx]
        .value_counts(normalize=True)
        .sort_index()
    )

    if not all(
        cls in test_class_prop.index
        for cls in overall_class_prop.index
    ):
        continue

    class_difference = np.sum(
        np.abs(
            overall_class_prop.values -
            test_class_prop.values
        )
    )

    test_fraction = len(test_idx) / len(X)

    size_difference = abs(
        test_fraction - 0.30
    )


    score = (
        class_difference +
        size_difference
    )

    candidate_splits.append({
        'train_idx': train_idx,
        'test_idx': test_idx,
        'test_fraction': test_fraction,
        'class_difference': class_difference,
        'score': score
    })


best_split = min(
    candidate_splits,
    key=lambda x: x['score']
)

train_idx = best_split['train_idx']
test_idx = best_split['test_idx']



X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_train = groups.iloc[train_idx].copy()
groups_test = groups.iloc[test_idx].copy()


print("\n TRAIN / TEST ")

print("\nTrain shape:")
print(X_train.shape)

print("\nTest shape:")
print(X_test.shape)


print("\nTrain class distribution:")
print(y_train.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())


print("\nTrain class proportions:")
print(y_train.value_counts(normalize=True))

print("\nTest class proportions:")
print(y_test.value_counts(normalize=True))


print("\nTrain time groups:")
print(sorted(groups_train.unique()))

print("\nTest time groups:")
print(sorted(groups_test.unique()))

time_overlap = set(
    groups_train.unique()
).intersection(
    set(groups_test.unique())
)

print("\nTime groups shared between train and test:")
print(time_overlap)

assert len(time_overlap) == 0, (
    "ERROR: The same time point occurs in both "
    "training and testing sets."
)

split_info = metadata.copy()

split_info['label'] = y

split_info['dataset'] = 'train'

split_info.loc[
    split_info.index.isin(X_test.index),
    'dataset'
] = 'test'

print(
    split_info[
        ['GEO_Accession', 'treatment', 'time', 'label', 'dataset']
    ].head()
)

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_splits = list(
    cv.split(
        X_train,
        y_train,
        groups=groups_train
    )
)

for fold, (cv_train_idx, cv_valid_idx) in enumerate(
    cv_splits,
    start=1
):

    cv_train_groups = groups_train.iloc[cv_train_idx]
    cv_valid_groups = groups_train.iloc[cv_valid_idx]

    cv_train_y = y_train.iloc[cv_train_idx]
    cv_valid_y = y_train.iloc[cv_valid_idx]

    print("\n====================================")
    print(f"Fold {fold}")
    print("====================================")
    print(
        "CV-train samples:",
        len(cv_train_idx)
    )
    print(
        "CV-validation samples:",
        len(cv_valid_idx)
    )

    print(
        "CV-train classes:"
    )
    print(cv_train_y.value_counts())

    print(
        "CV-validation classes:"
    )
    print(cv_valid_y.value_counts())

    print(
        "CV-train time groups:"
    )
    print(sorted(cv_train_groups.unique()))

    print(
        "CV-validation time groups:"
    )
    print(sorted(cv_valid_groups.unique())
    )

    # Critical temporal leakage check
    overlap = set(
        cv_train_groups.unique()
    ).intersection(
        set(cv_valid_groups.unique())
    )

    print(
        "Shared time groups:",
        overlap
    )

    assert len(overlap) == 0, (
        f"Temporal leakage detected in CV fold {fold}!"
    )

X_train = X_train.copy()
X_test = X_test.copy()

y_train = y_train.copy()
y_test = y_test.copy()

yeo_transformer = PowerTransformer(
    method='yeo-johnson',
    standardize=False
)

yeo_transformer.fit(X_train)

X_train_yeo = pd.DataFrame(
    yeo_transformer.transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_yeo = pd.DataFrame(
    yeo_transformer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


print("\nAfter Yeo-Johnson:")
print("Training shape:", X_train_yeo.shape)
print("Testing shape :", X_test_yeo.shape)

scaler = MinMaxScaler()
scaler.fit(X_train_yeo)

X_train_scaled = pd.DataFrame(
    scaler.transform(X_train_yeo),
    columns=X_train_yeo.columns,
    index=X_train_yeo.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_yeo),
    columns=X_test_yeo.columns,
    index=X_test_yeo.index
)

print("\nAfter Min-Max scaling:")
print("Training shape:", X_train_scaled.shape)
print("Testing shape :", X_test_scaled.shape)

zero_counts_train = (X_train_scaled == 0).sum()

zero_ratio_train = zero_counts_train / X_train_scaled.shape[0]

sparse_cols = (
    zero_ratio_train
    .sort_values(ascending=False)
    .head(10)
)

threshold = 0.9

mostly_zero_cols = zero_ratio_train[
    zero_ratio_train > threshold
].index

print(
    f"\nGenes removed based on >{threshold*100:.0f}% "
    f"zeros in training data: {len(mostly_zero_cols)}"
)
X_train_clean = X_train_scaled.drop(
    columns=mostly_zero_cols
)

X_test_clean = X_test_scaled.drop(
    columns=mostly_zero_cols
)

print("\nAfter zero filtering:")
print("Training shape:", X_train_clean.shape)
print("Testing shape :", X_test_clean.shape)

train_clean = X_train_clean.copy()
train_clean['label'] = y_train
test_clean = X_test_clean.copy()
test_clean['label'] = y_test

print("\nFinal datasets:")
print("Training:", train_clean.shape)
print("Testing :", test_clean.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("Original training shape:", X_train.shape)
print("Original testing shape :", X_test.shape)

print("Genes removed:", len(mostly_zero_cols))

print("Final training shape:", X_train_clean.shape)
print("Final testing shape :", X_test_clean.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# Stability Selection

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.base import clone

def group_stability_selection(
    X,
    y,
    groups,
    n_bootstraps=1000,
    sample_frac=0.75,
    n_features_to_select=10,
    C=0.5,
    random_state=42
):


    assert X.index.equals(y.index)
    assert X.index.equals(groups.index)

    assert groups.notna().all()
    assert y.nunique() == 2

    feature_names = X.columns.tolist()

    n_features = X.shape[1]

    unique_groups = groups.unique()

    n_groups = len(unique_groups)

    n_sampled_groups = max(
        2,
        int(np.floor(sample_frac * n_groups))
    )

    if n_sampled_groups > n_groups:
        raise ValueError("Invalid group subsampling fraction.")

    rng = np.random.default_rng(random_state)

    selection_counts = np.zeros(
        n_features,
        dtype=int
    )

    successful_iterations = 0

    for iteration in range(n_bootstraps):

        max_attempts = 1000

        for attempt in range(max_attempts):

            sampled_groups = rng.choice(
                unique_groups,
                size=n_sampled_groups,
                replace=False
            )

            mask = groups.isin(sampled_groups)

            X_sub = X.loc[mask]
            y_sub = y.loc[mask]

            if y_sub.nunique() == 2:
                break

        else:

            raise RuntimeError(
                "Unable to obtain a group subsample "
                "containing both classes."
            )

        l1_clf = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=C,
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state + iteration
        )

        l1_clf.fit(
            X_sub,
            y_sub
        )

        selected_mask = (
            np.abs(l1_clf.coef_.ravel()) > 1e-8
        )

        selection_counts += selected_mask.astype(int)

        successful_iterations += 1

    selection_freq = (
        selection_counts / successful_iterations
    )

    selection_results = pd.DataFrame({

        'Gene': feature_names,

        'Selection_Count': selection_counts,

        'Selection_Frequency': selection_freq

    })

    selection_results = selection_results.sort_values(
        by='Selection_Frequency',
        ascending=False
    ).reset_index(drop=True)

    selected_genes = selection_results.head(
        n_features_to_select
    )['Gene'].tolist()

    return selected_genes, selection_results

fold_selected_genes = {}
fold_selection_results = {}

for fold, (cv_train_idx, cv_valid_idx) in enumerate(cv_splits, start=1):

    print(f"\n{'='*60}")
    print(f"Fold {fold}")
    print(f"{'='*60}")

    X_fold_train = X_train.iloc[cv_train_idx].copy()
    y_fold_train = y_train.iloc[cv_train_idx].copy()
    groups_fold_train = groups_train.iloc[cv_train_idx].copy()

    groups_fold_valid = groups_train.iloc[cv_valid_idx].copy()

    overlap = set(groups_fold_train.unique()).intersection(
        set(groups_fold_valid.unique())
    )

    assert len(overlap) == 0, f"Group overlap detected in fold {fold}!"

    print(f"Training samples: {X_fold_train.shape[0]}")
    print(f"Training groups: {groups_fold_train.nunique()}")
    print(f"Validation groups: {groups_fold_valid.nunique()}")

    selected_genes_fold, selection_results_fold = group_stability_selection(
        X=X_fold_train,
        y=y_fold_train,
        groups=groups_fold_train,
        n_bootstraps=1000,
        sample_frac=0.75,
        n_features_to_select=1000,
        C=0.5,
        random_state=42 + fold
    )

    fold_selected_genes[fold] = selected_genes_fold

    fold_selection_results[fold] = selection_results_fold.copy()

    print(f"\nTop 10 selected genes in fold {fold}:")
    print(selection_results_fold.head(10))

    print(f"\nTotal genes recorded: {len(selection_results_fold)}")
    print(f"Top 1000 genes retained: {len(selected_genes_fold)}")

fold_gene_tables = []

for fold in range(1, 6):

    results = fold_selection_results[fold].copy()

    top_genes = results.head(1000).copy()

    top_genes['Selection_Rank'] = range(1, len(top_genes) + 1)

    top_genes['Fold'] = fold

    top_genes = top_genes[
        [
            'Gene',
            'Fold',
            'Selection_Rank',
            'Selection_Frequency'
        ]
    ]

    fold_gene_tables.append(top_genes)


all_fold_genes = pd.concat(
    fold_gene_tables,
    ignore_index=True
)

print("Total gene entries:", len(all_fold_genes))
print("Unique candidate genes:", all_fold_genes['Gene'].nunique())

all_fold_genes.head()


consensus_ranking = (
    all_fold_genes
    .groupby('Gene')
    .agg(
        Selection_Frequency_Count=('Fold', 'nunique'),
        Mean_Selection_Rank=('Selection_Rank', 'mean'),
        Mean_L1_Selection_Frequency=('Selection_Frequency', 'mean')
    )
    .reset_index()
)

consensus_ranking = consensus_ranking.sort_values(
    by=[
        'Selection_Frequency_Count',
        'Mean_Selection_Rank',
        'Mean_L1_Selection_Frequency'
    ],
    ascending=[False, True, False]
).reset_index(drop=True)

consensus_ranking.insert(
    0,
    'Consensus_Rank',
    range(1, len(consensus_ranking) + 1)
)

print("Total unique genes in consensus ranking:",
      len(consensus_ranking))

consensus_ranking.head(20)


final_gene_panel = consensus_ranking.head(10).copy()

final_genes = final_gene_panel['Gene'].tolist()

print("\nFinal 10-gene panel:")
print(final_gene_panel)

print("\nFinal gene IDs:")
print(final_genes)

# Boruta

from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

def group_boruta_selection(
    X,
    y,
    groups,
    n_bootstraps=1000,
    sample_frac=0.75,
    n_features_to_select=1000,
    n_estimators=200,
    max_iter=100,
    random_state=42
):

    assert X.index.equals(y.index)
    assert X.index.equals(groups.index)
    assert groups.notna().all()
    assert y.nunique() == 2

    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be between 0 and 1.")

    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be at least 1.")

    feature_names = X.columns.tolist()

    unique_groups = groups.unique()
    n_groups = len(unique_groups)

    n_sampled_groups = max(
        2,
        int(np.floor(sample_frac * n_groups))
    )

    if n_sampled_groups > n_groups:
        raise ValueError("Insufficient groups for the requested sampling.")

    rng = np.random.default_rng(random_state)

    confirmed_counts = np.zeros(len(feature_names), dtype=int)
    tentative_counts = np.zeros(len(feature_names), dtype=int)

    for iteration in range(n_bootstraps):

        for attempt in range(1000):

            sampled_groups = rng.choice(
                unique_groups,
                size=n_sampled_groups,
                replace=False
            )

            mask = groups.isin(sampled_groups)

            X_sub = X.loc[mask]
            y_sub = y.loc[mask]

            if y_sub.nunique() == 2:
                break

        else:
            raise RuntimeError(
                "Unable to obtain a group subsample containing both classes."
            )

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced',
            random_state=random_state + iteration,
            n_jobs=-1
        )

        boruta = BorutaPy(
            estimator=rf,
            n_estimators='auto',
            perc=100,
            alpha=0.05,
            two_step=True,
            max_iter=max_iter,
            random_state=random_state + iteration,
            verbose=0
        )

        boruta.fit(
            X_sub.to_numpy(),
            y_sub.to_numpy()
        )

        confirmed_counts += boruta.support_.astype(int)
        tentative_counts += boruta.support_weak_.astype(int)

    selection_results = pd.DataFrame({

        'Gene': feature_names,

        'Selection_Count': confirmed_counts,

        'Selection_Frequency': confirmed_counts / n_bootstraps,

        'Tentative_Count': tentative_counts,

        'Tentative_Frequency': tentative_counts / n_bootstraps

    })

    selection_results = selection_results.sort_values(
        by=[
            'Selection_Frequency',
            'Tentative_Frequency',
            'Gene'
        ],
        ascending=[False, False, True],
        kind='stable'
    ).reset_index(drop=True)

    selected_genes = selection_results.head(
        min(n_features_to_select, len(selection_results))
    )['Gene'].tolist()

    return selected_genes, selection_results

fold_boruta_selected_genes = {}
fold_boruta_results = {}

for fold, (cv_train_idx, cv_valid_idx) in enumerate(cv_splits, start=1):

    print(f"\n{'='*60}\nFold {fold}\n{'='*60}")

    X_fold_train = X_train.iloc[cv_train_idx].copy()
    y_fold_train = y_train.iloc[cv_train_idx].copy()
    groups_fold_train = groups_train.iloc[cv_train_idx].copy()

    groups_fold_valid = groups_train.iloc[cv_valid_idx].copy()

    assert set(groups_fold_train.unique()).isdisjoint(
        set(groups_fold_valid.unique())
    ), f"Group overlap detected in fold {fold}!"

    selected_genes_fold, selection_results_fold = group_boruta_selection(
        X=X_fold_train,
        y=y_fold_train,
        groups=groups_fold_train,
        n_bootstraps=1000,
        sample_frac=0.75,
        n_features_to_select=1000,
        n_estimators=200,
        max_iter=100,
        random_state=42 + fold
    )

    fold_boruta_selected_genes[fold] = selected_genes_fold
    fold_boruta_results[fold] = selection_results_fold.copy()

    print(f"Training samples: {len(X_fold_train)}")
    print(f"Training groups: {groups_fold_train.nunique()}")
    print(f"Total genes recorded: {len(selection_results_fold)}")
    print(f"Top genes retained: {len(selected_genes_fold)}")

    print("\nTop 10 genes:")
    print(selection_results_fold.head(10))

fold_gene_tables = []

for fold in range(1, 6):

    top_genes = fold_boruta_results[fold].head(1000).copy()

    top_genes['Selection_Rank'] = range(1, len(top_genes) + 1)
    top_genes['Fold'] = fold

    fold_gene_tables.append(
        top_genes[
            [
                'Gene',
                'Fold',
                'Selection_Rank',
                'Selection_Frequency',
                'Tentative_Frequency'
            ]
        ]
    )

all_fold_genes = pd.concat(
    fold_gene_tables,
    ignore_index=True
)

consensus_ranking = (
    all_fold_genes
    .groupby('Gene')
    .agg(
        Selection_Frequency_Count=('Fold', 'nunique'),
        Mean_Selection_Rank=('Selection_Rank', 'mean'),
        Mean_Confirmed_Frequency=('Selection_Frequency', 'mean'),
        Mean_Tentative_Frequency=('Tentative_Frequency', 'mean')
    )
    .reset_index()
)

consensus_ranking = consensus_ranking.sort_values(
    by=[
        'Selection_Frequency_Count',
        'Mean_Selection_Rank',
        'Mean_Confirmed_Frequency',
        'Gene'
    ],
    ascending=[False, True, False, True],
    kind='stable'
).reset_index(drop=True)

consensus_ranking.insert(
    0,
    'Consensus_Rank',
    range(1, len(consensus_ranking) + 1)
)

final_gene_panel = consensus_ranking.head(10).copy()

final_genes = final_gene_panel['Gene'].tolist()

print("\nFinal 10-gene Boruta panel:")
print(final_gene_panel)

# Recursive Feature Elimination with Cross Validation (RFECV)

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


def group_rfe_selection(
    X,
    y,
    groups,
    n_bootstraps=1000,
    sample_frac=0.75,
    n_features_to_select=1000,
    step=0.1,
    C=0.5,
    random_state=42
):

    assert X.index.equals(y.index)
    assert X.index.equals(groups.index)
    assert groups.notna().all()
    assert y.nunique() == 2

    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be between 0 and 1.")

    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be at least 1.")

    if not 1 <= n_features_to_select <= X.shape[1]:
        raise ValueError("Invalid number of features to select.")

    feature_names = X.columns.tolist()

    unique_groups = groups.unique()
    n_groups = len(unique_groups)

    n_sampled_groups = max(
        2,
        int(np.floor(sample_frac * n_groups))
    )

    if n_sampled_groups > n_groups:
        raise ValueError("Insufficient groups for the requested sampling.")

    rng = np.random.default_rng(random_state)

    selection_counts = np.zeros(len(feature_names), dtype=int)
    ranking_sums = np.zeros(len(feature_names), dtype=float)

    for iteration in range(n_bootstraps):

        for attempt in range(1000):

            sampled_groups = rng.choice(
                unique_groups,
                size=n_sampled_groups,
                replace=False
            )

            mask = groups.isin(sampled_groups)

            X_sub = X.loc[mask]
            y_sub = y.loc[mask]

            if y_sub.nunique() == 2:
                break

        else:
            raise RuntimeError(
                "Unable to obtain a group subsample containing both classes."
            )

        estimator = LogisticRegression(
            penalty='l2',
            solver='liblinear',
            C=C,
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state + iteration
        )

        rfe = RFE(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=step,
            verbose=0
        )

        rfe.fit(X_sub, y_sub)

        selection_counts += rfe.support_.astype(int)
        ranking_sums += rfe.ranking_.astype(float)

    selection_results = pd.DataFrame({

        'Gene': feature_names,

        'Selection_Count': selection_counts,

        'Selection_Frequency': selection_counts / n_bootstraps,

        'Mean_RFE_Rank': ranking_sums / n_bootstraps

    })

    selection_results = selection_results.sort_values(
        by=[
            'Selection_Frequency',
            'Mean_RFE_Rank',
            'Gene'
        ],
        ascending=[False, True, True],
        kind='stable'
    ).reset_index(drop=True)

    selected_genes = selection_results.head(
        min(n_features_to_select, len(selection_results))
    )['Gene'].tolist()

    return selected_genes, selection_results

fold_rfe_selected_genes = {}
fold_rfe_results = {}

for fold, (cv_train_idx, cv_valid_idx) in enumerate(cv_splits, start=1):

    print(f"\n{'='*60}\nFold {fold}\n{'='*60}")

    X_fold_train = X_train.iloc[cv_train_idx].copy()
    y_fold_train = y_train.iloc[cv_train_idx].copy()
    groups_fold_train = groups_train.iloc[cv_train_idx].copy()

    groups_fold_valid = groups_train.iloc[cv_valid_idx].copy()

    assert set(groups_fold_train.unique()).isdisjoint(
        set(groups_fold_valid.unique())
    ), f"Group overlap detected in fold {fold}!"

    selected_genes_fold, selection_results_fold = group_rfe_selection(
        X=X_fold_train,
        y=y_fold_train,
        groups=groups_fold_train,
        n_bootstraps=1000,
        sample_frac=0.75,
        n_features_to_select=1000,
        step=0.1,
        C=0.5,
        random_state=42 + fold
    )

    fold_rfe_selected_genes[fold] = selected_genes_fold
    fold_rfe_results[fold] = selection_results_fold.copy()

    print(f"Training samples: {len(X_fold_train)}")
    print(f"Training groups: {groups_fold_train.nunique()}")
    print(f"Total genes recorded: {len(selection_results_fold)}")
    print(f"Top genes retained: {len(selected_genes_fold)}")

    print("\nTop 10 genes:")
    print(selection_results_fold.head(10))

fold_gene_tables = []

for fold in range(1, 6):

    top_genes = fold_rfe_results[fold].head(1000).copy()

    top_genes['Selection_Rank'] = range(1, len(top_genes) + 1)
    top_genes['Fold'] = fold

    fold_gene_tables.append(
        top_genes[
            [
                'Gene',
                'Fold',
                'Selection_Rank',
                'Selection_Frequency',
                'Mean_RFE_Rank'
            ]
        ]
    )

all_fold_genes = pd.concat(
    fold_gene_tables,
    ignore_index=True
)

consensus_ranking = (
    all_fold_genes
    .groupby('Gene')
    .agg(
        Selection_Frequency_Count=('Fold', 'nunique'),
        Mean_Selection_Rank=('Selection_Rank', 'mean'),
        Mean_Selection_Frequency=('Selection_Frequency', 'mean'),
        Mean_RFE_Rank=('Mean_RFE_Rank', 'mean')
    )
    .reset_index()
)

consensus_ranking = consensus_ranking.sort_values(
    by=[
        'Selection_Frequency_Count',
        'Mean_Selection_Rank',
        'Mean_Selection_Frequency',
        'Mean_RFE_Rank',
        'Gene'
    ],
    ascending=[False, True, False, True, True],
    kind='stable'
).reset_index(drop=True)

consensus_ranking.insert(
    0,
    'Consensus_Rank',
    range(1, len(consensus_ranking) + 1)
)

final_gene_panel = consensus_ranking.head(10).copy()

final_genes = final_gene_panel['Gene'].tolist()

print("\nFinal 10-gene RFE panel:")
print(final_gene_panel)

# Mutual Information- Minimum Redundancy Maximum Relevance (MImRMR)

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score


def calculate_pairwise_mi(X_disc, n_bins=3):


    n_samples, n_features = X_disc.shape

    one_hot = np.stack(
        [(X_disc == b).astype(np.float32) for b in range(n_bins)],
        axis=0
    )

    category_counts = [
        one_hot[b].sum(axis=0) for b in range(n_bins)
    ]

    pairwise_mi = np.zeros(
        (n_features, n_features),
        dtype=np.float32
    )

    for a in range(n_bins):
        for b in range(n_bins):

            joint_counts = one_hot[a].T @ one_hot[b]

            expected_counts = (
                category_counts[a][:, None] *
                category_counts[b][None, :]
            )

            valid = joint_counts > 0

            pairwise_mi[valid] += (
                joint_counts[valid] / n_samples
            ) * np.log(
                (joint_counts[valid] * n_samples) /
                expected_counts[valid]
            )

    return (pairwise_mi + pairwise_mi.T) / 2


def group_mrmr_selection(
    X,
    y,
    groups,
    n_bootstraps=1000,
    sample_frac=0.75,
    candidate_pool_size=2000,
    n_features_to_select=1000,
    n_bins=3,
    random_state=42
):

    


    if not (X.index.equals(y.index) and X.index.equals(groups.index)):
        raise ValueError("X, y, and groups must have identical indices.")

    if not X.columns.is_unique:
        raise ValueError("Gene names in X must be unique.")

    if groups.isna().any():
        raise ValueError("Groups contain missing values.")

    if y.nunique() != 2:
        raise ValueError("This implementation requires binary labels.")

    if not (0 < sample_frac <= 1):
        raise ValueError("sample_frac must be between 0 and 1.")

    feature_names = X.columns.to_numpy()
    n_features = len(feature_names)

    if candidate_pool_size < n_features_to_select:
        raise ValueError(
            "candidate_pool_size must be >= n_features_to_select."
        )

    if candidate_pool_size > n_features:
        raise ValueError(
            "candidate_pool_size cannot exceed the number of genes."
        )

    X_values = X.to_numpy(dtype=np.float32)
    y_values = y.to_numpy()
    groups_values = groups.to_numpy()

    if not np.isfinite(X_values).all():
        raise ValueError("X contains missing or infinite values.")

    unique_groups = np.unique(groups_values)

    if len(unique_groups) < 2:
        raise ValueError("At least two groups are required.")

    rng = np.random.default_rng(random_state)

    selection_counts = np.zeros(n_features, dtype=int)
    rank_sums = np.zeros(n_features, dtype=float)
    relevance_sums = np.zeros(n_features, dtype=float)

    valid_iterations = 0

    for iteration in range(n_bootstraps):

        n_sampled_groups = max(
            1, int(np.ceil(sample_frac * len(unique_groups)))
        )

        for attempt in range(100):

            sampled_groups = rng.choice(
                unique_groups,
                size=n_sampled_groups,
                replace=False
            )

            sample_mask = np.isin(
                groups_values,
                sampled_groups
            )

            X_sub = X_values[sample_mask]
            y_sub = y_values[sample_mask]

            if len(np.unique(y_sub)) == 2:
                break

        else:
            continue

        

        continuous_mi = mutual_info_classif(
            X_sub,
            y_sub,
            discrete_features=False,
            random_state=random_state + iteration
        )

        candidate_idx = np.argsort(
            continuous_mi
        )[-candidate_pool_size:][::-1]

        X_candidates = X_sub[:, candidate_idx]


        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode="ordinal",
            strategy="quantile"
        )

        X_disc = discretizer.fit_transform(
            X_candidates
        ).astype(np.int8)

        valid_features = np.ptp(X_disc, axis=0) > 0

        X_disc = X_disc[:, valid_features]
        candidate_idx = candidate_idx[valid_features]

        if X_disc.shape[1] < n_features_to_select:
            continue

        y_encoded = pd.factorize(y_sub)[0]

        relevance = np.array([
            mutual_info_score(y_encoded, X_disc[:, j])
            for j in range(X_disc.shape[1])
        ])

        pairwise_mi = calculate_pairwise_mi(
            X_disc,
            n_bins=n_bins
        )


        n_candidates = X_disc.shape[1]

        selected_mask = np.zeros(
            n_candidates,
            dtype=bool
        )

        redundancy_sum = np.zeros(
            n_candidates,
            dtype=float
        )

        selected_order = []

        for rank in range(n_features_to_select):

            scores = relevance - (
                redundancy_sum / max(rank, 1)
            )

            scores[selected_mask] = -np.inf

            best_idx = np.argmax(scores)

            selected_mask[best_idx] = True
            selected_order.append(best_idx)

            
            redundancy_sum += pairwise_mi[:, best_idx]

        selected_global_idx = candidate_idx[
            selected_order
        ]

        selection_counts[selected_global_idx] += 1

        rank_sums[selected_global_idx] += np.arange(
            1, n_features_to_select + 1
        )

        relevance_sums[selected_global_idx] += relevance[
            selected_order
        ]

        valid_iterations += 1


    if valid_iterations == 0:
        raise RuntimeError(
            "No valid subsampling iterations were completed."
        )

    selection_frequency = (
        selection_counts / valid_iterations
    )

    mean_mrmr_rank = np.divide(
        rank_sums,
        selection_counts,
        out=np.full(n_features, np.nan),
        where=selection_counts > 0
    )

    mean_mi_relevance = np.divide(
        relevance_sums,
        selection_counts,
        out=np.full(n_features, np.nan),
        where=selection_counts > 0
    )

    results_df = pd.DataFrame({
        "Gene": feature_names,
        "Selection_Count": selection_counts,
        "Selection_Frequency": selection_frequency,
        "Mean_MRMR_Rank": mean_mrmr_rank,
        "Mean_MI_Relevance": mean_mi_relevance
    })

    results_df = results_df.sort_values(
        by=[
            "Selection_Frequency",
            "Mean_MRMR_Rank",
            "Mean_MI_Relevance",
            "Gene"
        ],
        ascending=[False, True, False, True],
        na_position="last"
    ).reset_index(drop=True)

    selected_genes = results_df.head(
        n_features_to_select
    )["Gene"].tolist()

    return selected_genes, results_df


fold_mrmr_selected_genes = {}
fold_mrmr_results = {}

for fold, (cv_train_idx, cv_valid_idx) in enumerate(
    cv_splits, start=1
):

    print(f"\nProcessing fold {fold}...")

    X_fold_train = X_train.iloc[cv_train_idx].copy()
    y_fold_train = y_train.iloc[cv_train_idx].copy()
    groups_fold_train = groups_train.iloc[cv_train_idx].copy()

    groups_fold_valid = groups_train.iloc[cv_valid_idx].copy()

    assert set(groups_fold_train.unique()).isdisjoint(
        set(groups_fold_valid.unique())
    ), f"Group overlap detected in fold {fold}!"

    selected_genes_fold, results_fold = group_mrmr_selection(
        X=X_fold_train,
        y=y_fold_train,
        groups=groups_fold_train,
        n_bootstraps=1000,
        sample_frac=0.75,
        candidate_pool_size=2000,
        n_features_to_select=1000,
        n_bins=3,
        random_state=42 + fold
    )

    fold_mrmr_selected_genes[fold] = selected_genes_fold
    fold_mrmr_results[fold] = results_fold.copy()

    print(
        f"Fold {fold}: {len(selected_genes_fold)} genes selected."
    )

all_fold_results = []

for fold, results in fold_mrmr_results.items():

    temp = results.copy()
    temp["Fold"] = fold
    temp["Fold_Rank"] = np.arange(1, len(temp) + 1)
    temp["In_Top1000"] = temp["Fold_Rank"] <= 1000

    all_fold_results.append(temp)

combined_results = pd.concat(
    all_fold_results,
    ignore_index=True
)

consensus_mrmr = (
    combined_results
    .groupby("Gene")
    .agg(
        Selection_Frequency_Count=("In_Top1000", "sum"),
        Mean_Selection_Rank=("Fold_Rank", "mean"),
        Mean_Selection_Frequency=("Selection_Frequency", "mean"),
        Mean_MRMR_Rank=("Mean_MRMR_Rank", "mean"),
        Mean_MI_Relevance=("Mean_MI_Relevance", "mean")
    )
    .reset_index()
)

consensus_mrmr = consensus_mrmr.sort_values(
    by=[
        "Selection_Frequency_Count",
        "Mean_Selection_Rank",
        "Mean_Selection_Frequency",
        "Mean_MRMR_Rank",
        "Mean_MI_Relevance",
        "Gene"
    ],
    ascending=[False, True, False, True, False, True],
    na_position="last"
).reset_index(drop=True)

final_mrmr_genes = consensus_mrmr.head(10)["Gene"].tolist()

print("Final 10-gene MI-mRMR panel:")
print(final_mrmr_genes)


# Embedded gradient boosting feature selection (EGBFS)

from sklearn.ensemble import GradientBoostingClassifier

def group_egbfs_selection(
    X,
    y,
    groups,
    n_bootstraps=1000,
    sample_frac=0.75,
    n_features_to_select=1000,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    max_features="sqrt",
    random_state=42
):

    if not (X.index.equals(y.index) and X.index.equals(groups.index)):
        raise ValueError("X, y, and groups must have identical indices.")

    if not X.columns.is_unique:
        raise ValueError("Gene names in X must be unique.")

    if groups.isna().any():
        raise ValueError("Groups contain missing values.")

    if y.nunique() != 2:
        raise ValueError("This implementation requires binary labels.")

    if not (0 < sample_frac <= 1):
        raise ValueError("sample_frac must be between 0 and 1.")

    feature_names = X.columns.to_numpy()
    n_features = len(feature_names)

    if n_features_to_select > n_features:
        raise ValueError(
            "n_features_to_select cannot exceed the number of genes."
        )

    X_values = X.to_numpy(dtype=np.float32)
    y_values = y.to_numpy()
    groups_values = groups.to_numpy()

    if not np.isfinite(X_values).all():
        raise ValueError("X contains missing or infinite values.")

    unique_groups = np.unique(groups_values)

    if len(unique_groups) < 2:
        raise ValueError("At least two groups are required.")

    rng = np.random.default_rng(random_state)

    selection_counts = np.zeros(n_features, dtype=int)
    importance_sums = np.zeros(n_features, dtype=float)
    rank_sums = np.zeros(n_features, dtype=float)

    valid_iterations = 0

    for iteration in range(n_bootstraps):

        n_sampled_groups = max(
            1, int(np.ceil(sample_frac * len(unique_groups)))
        )

        for attempt in range(100):

            sampled_groups = rng.choice(
                unique_groups,
                size=n_sampled_groups,
                replace=False
            )

            sample_mask = np.isin(
                groups_values,
                sampled_groups
            )

            X_sub = X_values[sample_mask]
            y_sub = y_values[sample_mask]

            if len(np.unique(y_sub)) == 2:
                break

        else:
            continue

        gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_features=max_features,
            random_state=random_state + iteration
        )

        gb.fit(X_sub, y_sub)

        importances = gb.feature_importances_

        importance_sum = importances.sum()

        if importance_sum > 0:

            normalized_importances = (
                importances / importance_sum
            )

        else:
            normalized_importances = np.zeros(n_features)

        importance_sums += normalized_importances

        positive_idx = np.flatnonzero(
            normalized_importances > 0
        )

        ranked_idx = positive_idx[
            np.argsort(
                normalized_importances[positive_idx]
            )[::-1]
        ]

        selected_idx = ranked_idx[:n_features_to_select]

        selection_counts[selected_idx] += 1

        rank_sums[selected_idx] += np.arange(
            1, len(selected_idx) + 1
        )

        valid_iterations += 1

    if valid_iterations == 0:
        raise RuntimeError(
            "No valid subsampling iterations were completed."
        )

    selection_frequency = (
        selection_counts / valid_iterations
    )

    mean_importance = (
        importance_sums / valid_iterations
    )

    mean_selection_rank = np.divide(
        rank_sums,
        selection_counts,
        out=np.full(n_features, np.nan),
        where=selection_counts > 0
    )

    results_df = pd.DataFrame({
        "Gene": feature_names,
        "Selection_Count": selection_counts,
        "Selection_Frequency": selection_frequency,
        "Mean_Normalized_Importance": mean_importance,
        "Mean_Selection_Rank": mean_selection_rank
    })

    results_df = results_df.sort_values(
        by=[
            "Selection_Frequency",
            "Mean_Normalized_Importance",
            "Mean_Selection_Rank",
            "Gene"
        ],
        ascending=[False, False, True, True],
        na_position="last"
    ).reset_index(drop=True)

    selected_genes = results_df.head(
        n_features_to_select
    )["Gene"].tolist()

    return selected_genes, results_df
fold_egbfs_selected_genes = {}
fold_egbfs_results = {}

for fold, (cv_train_idx, cv_valid_idx) in enumerate(
    cv_splits, start=1
):

    print(f"\nProcessing fold {fold}...")

    X_fold_train = X_train.iloc[cv_train_idx].copy()
    y_fold_train = y_train.iloc[cv_train_idx].copy()
    groups_fold_train = groups_train.iloc[cv_train_idx].copy()

    groups_fold_valid = groups_train.iloc[cv_valid_idx].copy()

    assert set(groups_fold_train.unique()).isdisjoint(
        set(groups_fold_valid.unique())
    ), f"Group overlap detected in fold {fold}!"

    selected_genes_fold, results_fold = group_egbfs_selection(
        X=X_fold_train,
        y=y_fold_train,
        groups=groups_fold_train,
        n_bootstraps=1000,
        sample_frac=0.75,
        n_features_to_select=1000,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        max_features="sqrt",
        random_state=42 + fold
    )

    fold_egbfs_selected_genes[fold] = selected_genes_fold
    fold_egbfs_results[fold] = results_fold.copy()

    print(
        f"Fold {fold}: {len(selected_genes_fold)} genes ranked."
    )

all_fold_results = []

for fold, results in fold_egbfs_results.items():

    temp = results.copy()
    temp["Fold"] = fold
    temp["Fold_Rank"] = np.arange(1, len(temp) + 1)
    temp["In_Top1000"] = temp["Fold_Rank"] <= 1000

    all_fold_results.append(temp)

combined_egbfs_results = pd.concat(
    all_fold_results,
    ignore_index=True
)

consensus_egbfs = (
    combined_egbfs_results
    .groupby("Gene")
    .agg(
        Selection_Frequency_Count=("In_Top1000", "sum"),
        Mean_Selection_Frequency=("Selection_Frequency", "mean"),
        Mean_Normalized_Importance=(
            "Mean_Normalized_Importance", "mean"
        ),
        Mean_Selection_Rank=("Fold_Rank", "mean")
    )
    .reset_index()
)

consensus_egbfs = consensus_egbfs.sort_values(
    by=[
        "Selection_Frequency_Count",
        "Mean_Selection_Frequency",
        "Mean_Normalized_Importance",
        "Mean_Selection_Rank",
        "Gene"
    ],
    ascending=[False, False, False, True, True]
).reset_index(drop=True)

final_egbfs_genes = consensus_egbfs.head(10)["Gene"].tolist()

print("Final 10-gene EGBFS panel:")
print(final_egbfs_genes)

# Genetic Algorithm (GA)

from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from sklearn_genetic import GAFeatureSelectionCV
def group_ga_selection(
    X,
    y,
    groups,
    n_bootstraps=1000,
    sample_frac=0.75,
    candidate_pool_size=2000,
    max_features=15,
    population_size=80,
    generations=35,
    cv_splits=5,
    random_state=42
):

    if not (X.index.equals(y.index) and X.index.equals(groups.index)):
        raise ValueError("X, y, and groups must have identical indices.")

    if not X.columns.is_unique:
        raise ValueError("Gene names must be unique.")

    if groups.isna().any():
        raise ValueError("Groups contain missing values.")

    if y.nunique() != 2:
        raise ValueError("Binary class labels are required.")

    if not (0 < sample_frac <= 1):
        raise ValueError("sample_frac must be between 0 and 1.")

    feature_names = X.columns.to_numpy()
    n_features = len(feature_names)

    if candidate_pool_size > n_features:
        raise ValueError("Candidate pool exceeds the number of genes.")

    if max_features > candidate_pool_size:
        raise ValueError("max_features exceeds candidate pool size.")

    X_values = X.to_numpy(dtype=np.float32)
    y_values = y.to_numpy()
    groups_values = groups.to_numpy()

    if not np.isfinite(X_values).all():
        raise ValueError("X contains missing or infinite values.")

    unique_groups = np.unique(groups_values)

    rng = np.random.default_rng(random_state)

    selection_counts = np.zeros(n_features, dtype=int)
    valid_iterations = 0

    for iteration in range(n_bootstraps):

        n_sampled_groups = max(
            1, int(np.ceil(sample_frac * len(unique_groups)))
        )

        for attempt in range(100):

            sampled_groups = rng.choice(
                unique_groups,
                size=n_sampled_groups,
                replace=False
            )

            sample_mask = np.isin(
                groups_values,
                sampled_groups
            )

            X_sub = X_values[sample_mask]
            y_sub = y_values[sample_mask]
            groups_sub = groups_values[sample_mask]

            if len(np.unique(y_sub)) == 2:
                break

        else:
            continue

        mi_scores = mutual_info_classif(
            X_sub,
            y_sub,
            discrete_features=False,
            random_state=random_state + iteration
        )

        candidate_idx = np.argsort(
            mi_scores
        )[-candidate_pool_size:][::-1]

        X_candidates = X_sub[:, candidate_idx]
        candidate_names = feature_names[candidate_idx]

        n_groups_sub = len(np.unique(groups_sub))

        n_inner_splits = min(cv_splits, n_groups_sub)

        if n_inner_splits < 2:
            continue

        inner_cv = StratifiedGroupKFold(
            n_splits=n_inner_splits,
            shuffle=True,
            random_state=random_state + iteration
        )

        inner_splits = list(
            inner_cv.split(
                X_candidates,
                y_sub,
                groups_sub
            )
        )


        estimator = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            C=0.5,
            class_weight="balanced",
            random_state=random_state + iteration,
            max_iter=1000
        )

        ga = GAFeatureSelectionCV(
            estimator=estimator,
            cv=inner_splits,
            scoring="f1_macro",
            population_size=population_size,
            generations=generations,
            crossover_probability=0.5,
            mutation_probability=0.2,
            keep_top_k=10,
            n_jobs=-1,
            verbose=False,
            max_features=max_features
        )

        ga.fit(X_candidates, y_sub)

        selected_mask = ga.get_support()

        selected_idx = candidate_idx[selected_mask]

        selection_counts[selected_idx] += 1

        valid_iterations += 1

        print(
            f"Iteration {iteration + 1}/{n_bootstraps}: "
            f"{len(selected_idx)} genes selected."
        )

    if valid_iterations == 0:
        raise RuntimeError(
            "No valid GA iterations were completed."
        )

    selection_frequency = (
        selection_counts / valid_iterations
    )

    results_df = pd.DataFrame({
        "Gene": feature_names,
        "Selection_Count": selection_counts,
        "Selection_Frequency": selection_frequency
    })

    results_df = results_df.sort_values(
        by=["Selection_Frequency", "Gene"],
        ascending=[False, True]
    ).reset_index(drop=True)

    selected_genes = results_df.loc[
        results_df["Selection_Count"] > 0,
        "Gene"
    ].tolist()

    return selected_genes, results_df

fold_ga_selected_genes = {}
fold_ga_results = {}

for fold, (cv_train_idx, cv_valid_idx) in enumerate(
    cv_splits, start=1
):

    print(f"\nProcessing fold {fold}...")

    X_fold_train = X_train.iloc[cv_train_idx].copy()
    y_fold_train = y_train.iloc[cv_train_idx].copy()
    groups_fold_train = groups_train.iloc[cv_train_idx].copy()

    groups_fold_valid = groups_train.iloc[cv_valid_idx].copy()

    assert set(groups_fold_train.unique()).isdisjoint(
        set(groups_fold_valid.unique())
    ), f"Group overlap detected in fold {fold}!"

    selected_genes_fold, results_fold = group_ga_selection(
        X=X_fold_train,
        y=y_fold_train,
        groups=groups_fold_train,
        n_bootstraps=1000,          
        sample_frac=0.75,
        candidate_pool_size=2000,
        max_features=15,
        population_size=80,
        generations=35,
        cv_splits=5,
        random_state=42 + fold
    )

    fold_ga_selected_genes[fold] = selected_genes_fold
    fold_ga_results[fold] = results_fold.copy()

    print(
        f"Fold {fold}: "
        f"{len(selected_genes_fold)} unique genes selected."
    )

all_fold_results = []

for fold, results in fold_ga_results.items():

    temp = results.copy()
    temp["Fold"] = fold

    all_fold_results.append(temp)

combined_ga_results = pd.concat(
    all_fold_results,
    ignore_index=True
)

consensus_ga = (
    combined_ga_results
    .groupby("Gene")
    .agg(
        Selection_Frequency_Count=("Selection_Count", lambda x: (x > 0).sum()),
        Mean_Selection_Frequency=("Selection_Frequency", "mean"),
        Total_Selection_Count=("Selection_Count", "sum")
    )
    .reset_index()
)

consensus_ga = consensus_ga.sort_values(
    by=[
        "Selection_Frequency_Count",
        "Mean_Selection_Frequency",
        "Total_Selection_Count",
        "Gene"
    ],
    ascending=[False, False, False, True]
).reset_index(drop=True)

final_ga_genes = consensus_ga.head(10)["Gene"].tolist()

print("Final 10-gene GA panel:")
print(final_ga_genes)

#  MLPNN 

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import f1_score
import itertools
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

feature_subsets = {
    "Stability_sel": ['LOC_Os10g38000', 'LOC_Os05g11610', 'LOC_Os01g74140', 'LOC_Os03g13030',
                      'LOC_Os04g40874', 'LOC_Os01g62830', 'LOC_Os04g16130', 'LOC_Os10g41270',
                      'LOC_Os11g13840', 'LOC_Os10g36520'],
    "Boruta":        ['LOC_Os01g04330', 'LOC_Os01g04590', 'LOC_Os01g06590', 'LOC_Os01g12000',
                      'LOC_Os01g14100', 'LOC_Os01g15350', 'LOC_Os01g15640', 'LOC_Os01g18220',
                      'LOC_Os01g19170', 'LOC_Os01g22600'],
    "RFE":           ['LOC_Os01g11280', 'LOC_Os01g74140', 'LOC_Os03g33520', 'LOC_Os03g38210',
                      'LOC_Os03g44290', 'LOC_Os04g16130', 'LOC_Os05g11610', 'LOC_Os09g30250',
                      'LOC_Os10g38000', 'LOC_Os10g41270'],
    "MRMR":          ['LOC_Os03g14010', 'LOC_Os07g42280', 'LOC_Os06g14670', 'LOC_Os01g18220',
                      'LOC_Os08g37444', 'LOC_Os03g09170', 'LOC_Os03g26910', 'LOC_Os09g26920',
                      'LOC_Os04g55920', 'LOC_Os02g52780'],
    "Tree_based":    ['LOC_Os03g09170', 'LOC_Os05g35740', 'LOC_Os06g06880', 'LOC_Os09g23780',
                      'LOC_Os10g07229', 'LOC_Os12g31400', 'LOC_Os11g09230', 'LOC_Os01g12180',
                      'LOC_Os07g15880', 'LOC_Os08g04840'],
    "Genetic":       ['LOC_Os01g51920', 'LOC_Os03g12320', 'LOC_Os04g45810', 'LOC_Os05g32210',
                      'LOC_Os06g51220', 'LOC_Os07g35520', 'LOC_Os07g44610', 'LOC_Os08g35760',
                      'LOC_Os10g30820', 'LOC_Os02g51164']
} ## Drought matrix

feature_subsets = {
    "Stability_sel": ['LOC_Os03g33590', 'LOC_Os02g43410', 'LOC_Os09g37910', 'LOC_Os11g13980'],
    "MRMR":          ['LOC_Os03g14180', 'LOC_Os03g64210', 'LOC_Os04g44830', 'LOC_Os01g04370']
} ## Heat matrix

hidden_layer_sizes = []
for n_layers in range(1, 5):  # 1, 2, 3, 4 layers
    for combo in itertools.product([50, 100], repeat=n_layers):
        hidden_layer_sizes.append(combo)

param_grid = {
    'hidden_layer_sizes': hidden_layer_sizes,
    'alpha': [1e-4, 1e-3, 1e-2],
    'learning_rate_init': [1e-3, 1e-2],
    'activation': ['relu', 'tanh']
}

mlp_results = []

for name, genes in feature_subsets.items():
    X_tr_sub = X_train[genes].values
    y_tr = y_train.values

    mlp = MLPClassifier(max_iter=1000, random_state=42,  verbose=True)
    grid = GridSearchCV(
        mlp,
        param_grid,
        cv=cv_splits,
        scoring='f1_macro',
        n_jobs=-1,
        refit=True,
        verbose=2
    )

    grid.fit(X_tr_sub, y_tr)

    best_params = grid.best_params_
    best_cv_score = grid.best_score_

    best_mlp = grid.best_estimator_
    y_tr_pred = best_mlp.predict(X_tr_sub)
    train_score = f1_score(y_tr, y_tr_pred, average='macro')

    mlp_results.append({
        'Method': name,
        'Best Params': best_params,
        'Train F1': train_score,
        'CV F1': best_cv_score
    })

df_mlp = pd.DataFrame(mlp_results)

import ace_tools_open as tools; tools.display_dataframe_to_user(name="MLP Hyperparameter Tuning Results", dataframe=df_mlp)


reports = {}

for _, row in df_mlp.iterrows():
    method = row['Method']
    best_params = row['Best Params']
    genes = feature_subsets[method]

    clf = MLPClassifier(
        **best_params,
        max_iter=1000,
        random_state=42
    )

    X_tr_sub = X_train[genes].values
    clf.fit(X_tr_sub, y_train.values)

    y_tr_pred = clf.predict(X_tr_sub)
    X_te_sub = X_test[genes].values
    y_te_pred = clf.predict(X_te_sub)

    cm_train = confusion_matrix(y_train, y_tr_pred, labels=clf.classes_)
    cm_test  = confusion_matrix(y_test, y_te_pred,   labels=clf.classes_)

    cr_train = classification_report(y_train, y_tr_pred, target_names=clf.classes_, digits=3)
    cr_test  = classification_report(y_test,  y_te_pred,  target_names=clf.classes_, digits=3)

    reports[method] = {
        'confusion_matrix_train': cm_train,
        'classification_report_train': cr_train,
        'confusion_matrix_test': cm_test,
        'classification_report_test': cr_test
    }

for method, rep in reports.items():
    print(f"\n===========")
    print(f"\n= {method} =")
    print(f"\n===========")
    print("Train Confusion Matrix:")
    print(rep['confusion_matrix_train'])
    print("\nTrain Classification Report:")
    print(rep['classification_report_train'])
    print("\nTest Confusion Matrix:")
    print(rep['confusion_matrix_test'])
    print("\nTest Classification Report:")
    print(rep['classification_report_test'])


# SVM 

import ast
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


param_grid = [
    {
        'kernel': ['linear'],
        'C': [0.01, 0.1, 1, 10, 100],
        'class_weight': [None, 'balanced']
    },

    {
        'kernel': ['rbf'],
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'class_weight': [None, 'balanced']
    },

    {
        'kernel': ['poly'],
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'degree': [2, 3],
        'class_weight': [None, 'balanced']
    }
]


svm_results = []

for name, genes in feature_subsets.items():

    print("\n" + "=" * 80)
    print(f"Running SVM for: {name}")
    print("=" * 80)

    X_tr_sub = X_train[genes].values
    y_tr = y_train.values

    svm = SVC(
        random_state=42
    )

    grid = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=cv_splits,
        scoring='f1_macro',
        n_jobs=-1,
        refit=True,
        verbose=2
    )

    grid.fit(X_tr_sub, y_tr)

    best_params = grid.best_params_

    best_cv_score = grid.best_score_

    
    best_svm = grid.best_estimator_

    y_tr_pred = best_svm.predict(X_tr_sub)

    train_accuracy = accuracy_score(y_tr, y_tr_pred)
    train_precision = precision_score(
        y_tr, y_tr_pred, average='macro', zero_division=0
    )
    train_recall = recall_score(
        y_tr, y_tr_pred, average='macro', zero_division=0
    )
    train_f1 = f1_score(
        y_tr, y_tr_pred, average='macro'
    )

    svm_results.append({
        'Method': name,
        'Best Params': best_params,
        'Train Accuracy': train_accuracy,
        'Train Precision': train_precision,
        'Train Recall': train_recall,
        'Train F1': train_f1,
        'CV F1': best_cv_score
    })

df_svm = pd.DataFrame(svm_results)

print("\n\nSVM HYPERPARAMETER TUNING RESULTS")
print("=" * 100)
print(df_svm.to_string(index=False))


svm_reports = {}

for _, row in df_svm.iterrows():

    method = row['Method']
    best_params = row['Best Params']

    genes = feature_subsets[method]

    print("\n" + "=" * 80)
    print(f"FINAL SVM EVALUATION: {method}")
    print("=" * 80)

    clf = SVC(
        **best_params,
        random_state=42
    )

    X_tr_sub = X_train[genes].values
    y_tr = y_train.values

    X_te_sub = X_test[genes].values
    y_te = y_test.values

    clf.fit(X_tr_sub, y_tr)

    y_tr_pred = clf.predict(X_tr_sub)
    y_te_pred = clf.predict(X_te_sub)

    train_accuracy = accuracy_score(y_tr, y_tr_pred)
    train_precision = precision_score(
        y_tr, y_tr_pred, average='macro', zero_division=0
    )
    train_recall = recall_score(
        y_tr, y_tr_pred, average='macro', zero_division=0
    )
    train_f1 = f1_score(
        y_tr, y_tr_pred, average='macro'
    )

    test_accuracy = accuracy_score(y_te, y_te_pred)
    test_precision = precision_score(
        y_te, y_te_pred, average='macro', zero_division=0
    )
    test_recall = recall_score(
        y_te, y_te_pred, average='macro', zero_division=0
    )
    test_f1 = f1_score(
        y_te, y_te_pred, average='macro'
    )

    cm_train = confusion_matrix(
        y_tr,
        y_tr_pred,
        labels=clf.classes_
    )

    cm_test = confusion_matrix(
        y_te,
        y_te_pred,
        labels=clf.classes_
    )

    cr_train = classification_report(
        y_tr,
        y_tr_pred,
        target_names=clf.classes_,
        digits=3,
        zero_division=0
    )

    cr_test = classification_report(
        y_te,
        y_te_pred,
        target_names=clf.classes_,
        digits=3,
        zero_division=0
    )

    svm_reports[method] = {

        'best_params': best_params,

        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,

        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,

        'confusion_matrix_train': cm_train,
        'confusion_matrix_test': cm_test,

        'classification_report_train': cr_train,
        'classification_report_test': cr_test
    }

for method, rep in svm_reports.items():

    print("\n")
    print("=" * 80)
    print(f"= {method} =")
    print("=" * 80)

    print("\nBest parameters:")
    print(rep['best_params'])

    print("\nTRAINING PERFORMANCE")
    print("-" * 50)
    print(f"Accuracy : {rep['train_accuracy']:.3f}")
    print(f"Precision: {rep['train_precision']:.3f}")
    print(f"Recall   : {rep['train_recall']:.3f}")
    print(f"F1       : {rep['train_f1']:.3f}")

    print("\nTEST PERFORMANCE")
    print("-" * 50)
    print(f"Accuracy : {rep['test_accuracy']:.3f}")
    print(f"Precision: {rep['test_precision']:.3f}")
    print(f"Recall   : {rep['test_recall']:.3f}")
    print(f"F1       : {rep['test_f1']:.3f}")

    print("\nTrain Confusion Matrix:")
    print(rep['confusion_matrix_train'])

    print("\nTest Confusion Matrix:")
    print(rep['confusion_matrix_test'])

    print("\nTrain Classification Report:")
    print(rep['classification_report_train'])

    print("Test Classification Report:")
    print(rep['classification_report_test'])

svm_test_results = []

for method, rep in svm_reports.items():

    svm_test_results.append({
        'Method': method,
        'Accuracy': rep['test_accuracy'],
        'Precision': rep['test_precision'],
        'Recall': rep['test_recall'],
        'F1': rep['test_f1']
    })

df_svm_test = pd.DataFrame(svm_test_results)

print("\n\nFINAL SVM TEST RESULTS")
print("=" * 80)
print(df_svm_test.to_string(index=False))

