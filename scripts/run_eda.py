"""Create EDA visualizations for the supplied/augmented clinical-feature dataset."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid')


def main():
    df = pd.read_csv(DATA/'dataset.csv')
    seed = pd.read_csv(DATA/'seed_dataset.csv')
    features = ['age','bmi','blood_pressure','glucose','cholesterol','smoking']

    # 1. Label distribution: original seed and augmented dataset side-by-side.
    counts = pd.DataFrame({
        'Original 500': seed['label'].value_counts(normalize=True).sort_index(),
        'Augmented 5,000': df['label'].value_counts(normalize=True).sort_index()
    }).fillna(0)
    ax = counts.plot(kind='bar', figsize=(7.5,4.5), rot=0)
    ax.set_title('Target-label distribution: seed vs augmented dataset')
    ax.set_xlabel('Label (0 = negative, 1 = positive per supplied prompt)')
    ax.set_ylabel('Share of records')
    ax.legend(title='Dataset')
    plt.tight_layout(); plt.savefig(OUT/'01_label_distribution.png', dpi=180); plt.close()

    # 2. Feature distributions.
    fig, axes = plt.subplots(2,3,figsize=(12,7))
    for ax, col in zip(axes.flat, features):
        sns.histplot(data=df, x=col, hue='label', bins=25, element='step', stat='density',
                     common_norm=False, ax=ax, alpha=.25)
        ax.set_title(col.replace('_',' ').title())
    fig.suptitle('Feature distributions by binary label (augmented dataset)', y=1.02)
    fig.tight_layout(); fig.savefig(OUT/'02_feature_distributions.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    # 3. Boxplots of continuous variables grouped by label.
    cont = ['age','bmi','blood_pressure','glucose','cholesterol']
    long = df.melt(id_vars='label', value_vars=cont, var_name='feature', value_name='value')
    fig, ax = plt.subplots(figsize=(11,5))
    sns.boxplot(data=long, x='feature', y='value', hue='label', ax=ax)
    ax.set_title('Clinical-feature distributions by label')
    ax.set_xlabel('Feature'); ax.set_ylabel('Recorded value (units not specified in supplied file)')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout(); fig.savefig(OUT/'03_feature_boxplots.png', dpi=180); plt.close(fig)

    # 4. Correlation heatmap.
    corr = df[features+['label']].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0, square=True, ax=ax)
    ax.set_title('Pearson correlation matrix (descriptive only)')
    fig.tight_layout(); fig.savefig(OUT/'04_correlation_heatmap.png', dpi=180); plt.close(fig)

    # 5. Compare seed and augmented distribution to make synthetic augmentation visible.
    fig, axes = plt.subplots(2,3,figsize=(12,7))
    for ax, col in zip(axes.flat, features):
        sns.histplot(seed[col], ax=ax, label='Original 500', stat='density', element='step',
                     fill=False, bins=25, color='#4c72b0')
        sns.histplot(df[col], ax=ax, label='Augmented 5,000', stat='density', element='step',
                     fill=False, bins=25, color='#dd8452')
        ax.set_title(col.replace('_',' ').title())
        ax.legend(fontsize=8)
    fig.suptitle('Original seed vs augmented feature distributions', y=1.02)
    fig.tight_layout(); fig.savefig(OUT/'05_seed_vs_augmented.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    print('EDA completed. Figures saved to', OUT)
    for p in sorted(OUT.glob('*.png')): print(p.name)

if __name__ == '__main__':
    main()
