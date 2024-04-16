# Prevendo o valor do preço de automóveis baseado em suas características
## Preparação dos dados
# %%
### Importando bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
### Carregando os dados
dados = pd.read_csv("data.csv")

# %%
### Checando os dados
dados.info()
dados.head(2)

# %%
### Padronizando o nome das variáveis
dados.columns
dados.columns = dados.columns.str.lower().str.replace(" ","_")
dados.columns

# %%
### Padronizando os valores das variáveis categóricas
var_categoricas = list(dados.dtypes[dados.dtypes == "object"].index)

for col in var_categoricas:
    dados[col] = dados[col].str.lower().str.replace(" ","_")

dados.head(2)

# %%
### Cria função que checa a quantidade de valor únicos e os n primeiros valores únicos das variáveis
def check_unique(df: pd.DataFrame, n: int):
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} valores únicos.")
        print(f"Primeiros {n} valores:")
        print(df[col].unique()[:n],"\n")

check_unique(dados,5)
# %%
### Cria função que verifica o número de registros nulos por variável
def check_null_number(df : pd.DataFrame):
    return df.isna().sum()

check_null_number(dados)

# %%
### Cria função que verifica o percentual de registros nulos por variável
def check_null_percentage(df : pd.DataFrame):
    return (df.isna().sum()/df.count()).map('{:.2%}'.format)

check_null_percentage(dados)

# %%
### Removendo os registros com valores nulos
dados = dados.dropna()
check_null_percentage(dados)

# %%
### Checando quantidade de registros
dados.count()

## Análise exploratória
# %%
### Analisando a distribuição do preco do veículo
sns.histplot(dados.msrp,bins=10)
# Existem alguns valores na casa de milhoes que não permitem ter uma noção correta da distribuição,
# precisaremos filtrar o conjunto exibido no gráfico
# %%
### Gráfico ajustado
sns.histplot(dados.msrp[dados.msrp < 100000], bins = 50)
# %%
# O gráfico apresenta um distribuição assimétrica que pode ser otimizada através de uma expressão logarítimica,
# se aproximando da normal e se adequando melhor ao modelo. Ajustaremos a variável para, colocando seus valores
# numa escala logarítimica e plotaremos o gráfico novamente.

dados["msrp"] = np.log1p(dados["msrp"])
sns.histplot(dados.msrp[dados.msrp < 100000], bins = 50)



# %%
