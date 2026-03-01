import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def afficher_infos(df):
    """Affiche les informations de base du dataframe."""
    print("Shape :", df.shape)
    print("\nTypes :\n", df.dtypes)
    print("\nValeurs manquantes :\n", df.isnull().sum())
    print("\nStatistiques :\n", df.describe())


def tracer_correlation(df, output_path="reports/correlation_heatmap.png"):
    """Trace et sauvegarde la heatmap de corrélation."""
    num_cols = df.select_dtypes(include=np.number).columns
    plt.figure(figsize=(14, 10))
    sns.heatmap(df[num_cols].corr(), cmap='coolwarm', annot=False)
    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Heatmap sauvegardée : {output_path}")


def supprimer_features_inutiles(df):
    """Supprime les colonnes à variance nulle (une seule valeur unique)."""
    avant = df.shape[1]
    df = df.loc[:, df.nunique() > 1]
    apres = df.shape[1]
    print(f"Colonnes supprimées (variance nulle) : {avant - apres}")
    return df


def parser_dates(df, col='RegistrationDate'):
    """Parse une colonne de dates et extrait des features temporelles."""
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
    df['RegYear']    = df[col].dt.year
    df['RegMonth']   = df[col].dt.month
    df['RegDay']     = df[col].dt.day
    df['RegWeekday'] = df[col].dt.weekday
    df.drop(columns=[col], inplace=True)
    return df


def feature_engineering(df):
    """Crée de nouvelles features à partir des features existantes."""
    if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)

    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency'].replace(0, 1)

    if 'Recency' in df.columns and 'CustomerTenure' in df.columns:
        df['TenureRatio'] = df['Recency'] / df['CustomerTenure'].replace(0, 1)

    return df


def corriger_aberrants(df):
    """Corrige les valeurs aberrantes connues."""
    # SupportTickets : -1 et 999 sont des codes d'erreur
    if 'SupportTickets' in df.columns:
        df['SupportTickets'] = df['SupportTickets'].replace([-1, 999], np.nan)

    # Satisfaction : -1, 0 et 99 sont des codes d'erreur
    if 'Satisfaction' in df.columns:
        df['Satisfaction'] = df['Satisfaction'].replace([-1, 0, 99], np.nan)

    return df