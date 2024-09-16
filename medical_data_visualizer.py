import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")

# 2
height = df["height"] * df["height"]
df["BMI"] = (df["weight"] / height) * 10000
df['overweight'] = (df["BMI"] > 25).astype(int)

# 3
df.loc[df["cholesterol"] == 1, "cholesterol"] = 0 #0 = bueno
df.loc[df["cholesterol"] > 1, "cholesterol"] = 1 #1 = malo

df.loc[df["gluc"] == 1, "gluc"] = 0
df.loc[df["gluc"] > 1, "gluc"] = 1

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=["cardio"],
                    value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])
    # 6
    df_cat = df_cat.groupby(["cardio", "variable", "value"], as_index=False).size()
    
    # 7
    df_cat = df_cat.rename(columns={"size": "total"})

    # 8
    fig = sns.catplot(x="variable", y="total", hue="value", col="cardio",
                    data=df_cat, kind="bar", height=5, aspect=1)
    fig = fig.fig
    
    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"]) &
        (df["height"] >= df["height"].quantile(0.025)) &
        (df["height"] <= df["height"].quantile(0.975)) &
        (df["weight"] >= df["weight"].quantile(0.025)) &
        (df["weight"] <= df["weight"].quantile(0.975)) 
    ].drop(columns=["BMI"])

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(12,10))

    # 15
    ticks = np.arange(-0.1, 0.4, 0.08)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm", vmax=0.3, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.5, "ticks": ticks}, ax=ax)
    
    # 16
    fig.savefig('heatmap.png')
    return fig
