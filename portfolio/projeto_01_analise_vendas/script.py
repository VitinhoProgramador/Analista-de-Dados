import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar Dados
print("Carregando dados...")
# O caminho é relativo ao diretório de execução, que é a raiz do repositório
df = pd.read_csv('portfolio/projeto_01_analise_vendas/data/vendas_dataset.csv')

# Conversão da coluna Data para datetime e extração do mês para análise de sazonalidade
df['Data'] = pd.to_datetime(df['Data'])
df['Mes'] = df['Data'].dt.month

# 2. Análise Exploratória de Dados (EDA)

# Vendas Totais por Região
vendas_regiao = df.groupby('Regiao')['Receita'].sum().sort_values(ascending=False)
print("\nVendas Totais por Região:\n", vendas_regiao)

# Visualização da Receita por Região
plt.figure(figsize=(8, 5))
sns.barplot(x=vendas_regiao.index, y=vendas_regiao.values)
plt.title('Receita Total por Região')
plt.ylabel('Receita (R$)')
plt.xlabel('Região')
# Salvando a imagem na pasta do projeto
plt.savefig('portfolio/projeto_01_analise_vendas/receita_por_regiao.png')
# plt.show() # Comentado para evitar erro em ambiente não-gráfico

# 3. Modelagem Preditiva (Regressão Linear Simples)

# Objetivo: Prever a Receita com base na Quantidade
X = df[['Quantidade']]
y = df['Receita']

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinamento do modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Previsões
y_pred = model.predict(X_test)

# Avaliação do modelo
r2 = r2_score(y_test, y_pred)
print(f"\nCoeficiente de Determinação (R²): {r2:.2f}")

# Exibir os coeficientes
print(f"Intercepto: {model.intercept_:.2f}")
print(f"Coeficiente (Quantidade): {model.coef_[0]:.2f}")

# 4. Conclusão
print("\nAnálise e modelagem concluídas. O script demonstra as habilidades de manipulação, visualização e modelagem em Python.")
