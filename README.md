
# Análise e Modelagem de Dados de Diabetes

Este projeto visa realizar a análise e modelagem de dados sobre diabetes utilizando diversos algoritmos de machine learning. O objetivo é construir modelos de classificação, avaliar suas performances e visualizar as métricas associadas.

## Desafios

### Desafio 1: Leitura e Preparação dos Dados

- **Objetivo:** Ler a base de dados sobre diabetes e separar as variáveis explicativas e a variável alvo.
- **Código:**
  ```python
  import pandas as pd

  # Carregar o dataset
  url = "https://raw.githubusercontent.com/lorranmendes22/Diabetes/main/diabetes.csv"
  dados = pd.read_csv(url)

  # Separar as variáveis explicativas e a variável alvo
  X = dados.drop('diabetes', axis=1)
  y = dados['diabetes']
  ```

### Desafio 2: Divisão dos Dados

- **Objetivo:** Dividir os dados em conjuntos de treino e teste.
- **Código:**
  ```python
  from sklearn.model_selection import train_test_split

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```

### Desafio 3: Criação e Avaliação dos Modelos

- **Objetivo:** Criar e avaliar os modelos de Decision Tree e Random Forest.
- **Código:**
  ```python
  from sklearn.tree import DecisionTreeClassifier, plot_tree
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score

  # Decision Tree
  dtc = DecisionTreeClassifier(max_depth=5)
  dtc.fit(X_train, y_train)
  y_train_pred = dtc.predict(X_train)
  y_test_pred = dtc.predict(X_test)
  train_accuracy = accuracy_score(y_train, y_train_pred)
  test_accuracy = accuracy_score(y_test, y_test_pred)

  # Random Forest
  rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
  rfc.fit(X_train, y_train)
  y_train_pred_rfc = rfc.predict(X_train)
  y_test_pred_rfc = rfc.predict(X_test)
  train_accuracy_rfc = accuracy_score(y_train, y_train_pred_rfc)
  test_accuracy_rfc = accuracy_score(y_test, y_test_pred_rfc)
  ```

### Desafio 4: Matriz de Confusão

- **Objetivo:** Construir e visualizar a matriz de confusão para os modelos.
- **Código:**
  ```python
  from sklearn.metrics import confusion_matrix
  import seaborn as sns
  import matplotlib.pyplot as plt

  # Decision Tree
  dtc_confusion_matrix = confusion_matrix(y_test, y_test_pred)
  sns.heatmap(dtc_confusion_matrix, annot=True, fmt="d", cmap="Blues")
  plt.title("Decision Tree Confusion Matrix")
  plt.xlabel("Predicted Labels")
  plt.ylabel("True Labels")
  plt.show()

  # Random Forest
  rfc_confusion_matrix = confusion_matrix(y_test, y_test_pred_rfc)
  sns.heatmap(rfc_confusion_matrix, annot=True, fmt="d", cmap="Blues")
  plt.title("Random Forest Confusion Matrix")
  plt.xlabel("Predicted Labels")
  plt.ylabel("True Labels")
  plt.show()
  ```

### Desafio 5: Métricas de Desempenho

- **Objetivo:** Extrair as métricas de acurácia, recall, precisão e F1-Score para os modelos.
- **Código:**
  ```python
  from sklearn.metrics import recall_score, precision_score, f1_score

  dtc_recall = recall_score(y_test, y_test_pred, average='weighted')
  dtc_precision = precision_score(y_test, y_test_pred, average='weighted')
  dtc_f1 = f1_score(y_test, y_test_pred, average='weighted')

  rfc_recall = recall_score(y_test, y_test_pred_rfc, average='weighted')
  rfc_precision = precision_score(y_test, y_test_pred_rfc, average='weighted')
  rfc_f1 = f1_score(y_test, y_test_pred_rfc, average='weighted')

  print(f"Decision Tree - Acurácia de Treino: {train_accuracy:.4f}")
  print(f"Decision Tree - Acurácia de Teste: {test_accuracy:.4f}")
  print(f"Random Forest - Acurácia de Treino: {train_accuracy_rfc:.4f}")
  print(f"Random Forest - Acurácia de Teste: {test_accuracy_rfc:.4f}")
  print(f"Decision Tree - Recall: {dtc_recall:.4f}")
  print(f"Decision Tree - Precisão: {dtc_precision:.4f}")
  print(f"Decision Tree - F1-Score: {dtc_f1:.4f}")
  print(f"Random Forest - Recall: {rfc_recall:.4f}")
  print(f"Random Forest - Precisão: {rfc_precision:.4f}")
  print(f"Random Forest - F1-Score: {rfc_f1:.4f}")
  ```

### Desafio 6: Curvas ROC

- **Objetivo:** Obter e comparar as curvas ROC dos modelos.
- **Código:**
  ```python
  from sklearn.metrics import RocCurveDisplay

  # Gerar dados de classificação
  from sklearn.datasets import make_classification
  from sklearn.model_selection import train_test_split
  from sklearn.svm import SVC

  # Gerar dados fictícios para exemplo
  X, y = make_classification(random_state=0)
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

  # Instanciar e treinar os modelos
  svc = SVC(random_state=42, probability=True)
  svc.fit(X_train, y_train)
  rfc = RandomForestClassifier(random_state=42)
  rfc.fit(X_train, y_train)

  # Exibir as curvas ROC no mesmo gráfico
  svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)
  rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=svc_disp.ax_)
  plt.title("Comparação das Curvas ROC")
  plt.show()
  ```

### Desafio 7: Curvas Precisão x Recall

- **Objetivo:** Obter e comparar as curvas precisão x recall dos modelos.
- **Código:**
  ```python
  from sklearn.metrics import PrecisionRecallDisplay

  # Calcular e exibir curvas precisão x recall
  svc_disp = PrecisionRecallDisplay.from_estimator(svc, X_test, y_test)
  rfc_disp = PrecisionRecallDisplay.from_estimator(rfc, X_test, y_test, ax=svc_disp.ax_)
  rfc_disp.figure_.suptitle("Precision-Recall curve comparison")
  plt.show()
  ```

## Requisitos

- Python 3.6 ou superior
- Bibliotecas: pandas, scikit-learn, matplotlib, seaborn

## Execução

1. Clone este repositório.
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute o script de análise e modelagem.

