import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
# Connexion à MetaTrader 5
if not mt5.initialize():
    print("MetaTrader 5 n'a pas pu être démarré")
    mt5.shutdown()

# Définir le symbole et la période
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_D1  # Données journalières

# Définir la plage de dates
start_date = datetime(2020, 9, 10)
end_date = datetime(2024, 6, 29)

# Télécharger les données pour la plage de dates spécifiée
rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)


# Convertir en DataFrame pandas
data = pd.DataFrame(rates)
data['time'] = pd.to_datetime(data['time'], unit='s')

# Afficher les premières lignes des données
print(data)

# Sauvegarder les données dans un fichier CSV
data.to_csv('data_analyzed1.csv', index=False)
# Extraire seulement les colonnes OHLC
data = data[['time','open', 'high', 'low', 'close']]
data_ohlc = data[['open', 'high', 'low', 'close']]

# Appliquer la mise en échelle Min-Max
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_ohlc)
scaled_data = pd.DataFrame(scaled_data, index=data_ohlc.index, columns=data_ohlc.columns)

print(data)
# Calculer les retours
data['Return'] = data['close'].pct_change()

# Fonction pour calculer la fraction de Kelly
def calculate_kelly_fraction(win_prob, win_loss_ratio):
    return (win_prob * (win_loss_ratio + 1) - 1) / win_loss_ratio

# Initialiser les paramètres de Kelly variables
rolling_window = 30  # Fenêtre de temps pour le calcul des probabilités
kelly_fractions = []

for i in range(len(data)):
    if i < rolling_window:
        kelly_fractions.append(0)  # Pas assez de données pour calculer
    else:
        window_data = data.iloc[i - rolling_window:i]
        wins = window_data[window_data['Return'] > 0]
        losses = window_data[window_data['Return'] <= 0]
        win_prob = len(wins) / rolling_window
        avg_win = wins['Return'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['Return'].mean()) if len(losses) > 0 else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        kelly_fraction = calculate_kelly_fraction(win_prob, win_loss_ratio) if win_loss_ratio > 0 else 0
        kelly_fractions.append(kelly_fraction)
# Afficher les paramètres de Kelly
print(f"Ligne {1000} : win_prob={win_prob:.2f}, avg_win={avg_win:.4f}, avg_loss={avg_loss:.4f}, win_loss_ratio={win_loss_ratio:.2f}, kelly_fraction={kelly_fraction:.4f}")

data['Kelly Fraction'] = kelly_fractions

# Valeur actuelle de la fraction de Kelly
current_kelly_value = data['Kelly Fraction'].iloc[-1]
print(f"Valeur actuelle de la fraction de Kelly : {current_kelly_value}")

# Vérification de la valeur de Kelly pour le levier
if current_kelly_value < 0:
    print("La fraction de Kelly est négative. Il n'est pas conseillé d'investir. Le levier devrait être 0.")
else:
    # Calculer le levier recommandé si la fraction de Kelly est positive (facultatif)
    recommended_leverage = current_kelly_value  # Vous pouvez ajuster cette valeur en fonction de vos critères
    print(f"La fraction de Kelly est positive. Le levier recommandé est {recommended_leverage}.")

# Sauvegarder les données dans un fichier CSV
data.to_csv('data_analyzed1.csv', index=False)

# Graphique de l'évolution des fractions de Kelly
plt.figure(figsize=(14, 7))
plt.plot(data['time'], data['Kelly Fraction'], label='Fraction de Kelly')
plt.xlabel('Date')
plt.ylabel('Fraction de Kelly')
plt.title('Évolution optimale de la fraction de Kelly pour EURUSD')
plt.legend()
plt.grid(True)
plt.show()
print(data)
# Déconnexion de MetaTrader 5
mt5.shutdown()

# Recharger les données analysées précédemment
data = pd.read_csv('data_analyzed1.csv') 

# Initialiser les capitaux
initial_capital = 10000

# Niveau de levier à appliquer
leverage = 1  # Modifiez cette valeur manuellement pour tester différents niveaux de levier
results = []

# Simulation de trading avec le niveau de levier spécifié
capital_kelly = initial_capital
capital_kelly_values = [capital_kelly]
for i in range(1, len(data)):
    trade_return = data['Return'].iloc[i]
    kelly_fraction = data['Kelly Fraction'].iloc[i]
    if kelly_fraction > 0:  # Ne pas investir si la fraction de Kelly est négative ou zéro
        trade_kelly = trade_return * kelly_fraction * leverage
        capital_kelly *= (1 + trade_kelly)
    capital_kelly_values.append(capital_kelly)
results = capital_kelly_values

# Convertir la colonne 'time' en format de date si nécessaire
data['time'] = pd.to_datetime(data['time'])

# Graphique de l'évolution du capital avec le levier spécifié
plt.figure(figsize=(14, 7))
plt.plot(data['time'], results, label=f'Levier {leverage}')
plt.xlabel('Date')
plt.ylabel('Capital')
plt.title('Évolution du capital avec levier pour EURUSD')
plt.legend()
plt.grid(True)
plt.show()
# Connexion à MetaTrader 5
if not mt5.initialize():
    print("MetaTrader 5 n'a pas pu être démarré")
    mt5.shutdown()

# Définir le symbole et la période
symbol = "US500"
timeframe = mt5.TIMEFRAME_D1  # Données journalières

# Définir la plage de dates
start_date = datetime(2020, 8, 18)
end_date = datetime(2024, 6, 29)

# Télécharger les données pour la plage de dates spécifiée
rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

# Convertir en DataFrame pandas
data = pd.DataFrame(rates)
data['time'] = pd.to_datetime(data['time'], unit='s')

# Afficher les premières lignes des données
print(data)

# Sauvegarder les données dans un fichier CSV
data.to_csv('data_analyzed2.csv', index=False)
# Extraire seulement les colonnes OHLC
data = data[['time','open', 'high', 'low', 'close']]
data_ohlc = data[['open', 'high', 'low', 'close']]


# Appliquer la mise en échelle Min-Max
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_ohlc)
scaled_data = pd.DataFrame(scaled_data, index=data_ohlc.index, columns=data_ohlc.columns)

print(data)

# Calculer les retours
data['Return'] = data['close'].pct_change()
# Fonction pour calculer la fraction de Kelly
def calculate_kelly_fraction(win_prob, win_loss_ratio):
    return (win_prob * (win_loss_ratio + 1) - 1) / win_loss_ratio
# Initialiser les paramètres de Kelly variables
rolling_window = 30  # Fenêtre de temps pour le calcul des probabilités
kelly_fractions = []
for i in range(len(data)):
    if i < rolling_window:
        kelly_fractions.append(0)  # Pas assez de données pour calculer
    else:
        window_data = data.iloc[i - rolling_window:i]
        wins = window_data[window_data['Return'] > 0]
        losses = window_data[window_data['Return'] <= 0]
        win_prob = len(wins) / rolling_window
        avg_win = wins['Return'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['Return'].mean()) if len(losses) > 0 else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        kelly_fraction = calculate_kelly_fraction(win_prob, win_loss_ratio) if win_loss_ratio > 0 else 0
        kelly_fractions.append(kelly_fraction)
# Afficher les paramètres de Kelly
print(f"Ligne {1000} : win_prob={win_prob:.2f}, avg_win={avg_win:.4f}, avg_loss={avg_loss:.4f}, win_loss_ratio={win_loss_ratio:.2f}, kelly_fraction={kelly_fraction:.4f}")
data['Kelly Fraction'] = kelly_fractions
# Valeur actuelle de la fraction de Kelly
current_kelly_value = data['Kelly Fraction'].iloc[-1]
print(f"Valeur actuelle de la fraction de Kelly : {current_kelly_value}")
# Vérification de la valeur de Kelly pour le levier
if current_kelly_value < 0:
    print("La fraction de Kelly est négative. Il n'est pas conseillé d'investir. Le levier devrait être 0.")
else:
    # Calculer le levier recommandé si la fraction de Kelly est positive (facultatif)
    recommended_leverage = current_kelly_value  # Vous pouvez ajuster cette valeur en fonction de vos critères
    print(f"La fraction de Kelly est positive. Le levier recommandé est {recommended_leverage}.")
# Sauvegarder les données dans un fichier CSV
data.to_csv('data_analyzed2.csv', index=False)
# Graphique de l'évolution des fractions de Kelly
plt.figure(figsize=(14, 7))
plt.plot(data['time'], data['Kelly Fraction'], label='Fraction de Kelly')
plt.xlabel('Date')
plt.ylabel('Fraction de Kelly')
plt.title('Évolution optimale de la fraction de Kelly pour US500Cash')
plt.legend()
plt.grid(True)
plt.show()
# Déconnexion de MetaTrader 5
mt5.shutdown()
print(data)
# Recharger les données analysées précédemment
data = pd.read_csv('data_analyzed2.csv') 

# Initialiser les capitaux
initial_capital = 10000

# Niveau de levier à appliquer
leverage = 1  # Modifiez cette valeur manuellement pour tester différents niveaux de levier
results = []

# Simulation de trading avec le niveau de levier spécifié
capital_kelly = initial_capital
capital_kelly_values = [capital_kelly]
for i in range(1, len(data)):
    trade_return = data['Return'].iloc[i]
    kelly_fraction = data['Kelly Fraction'].iloc[i]
    if kelly_fraction > 0:  # Ne pas investir si la fraction de Kelly est négative ou zéro
        trade_kelly = trade_return * kelly_fraction * leverage
        capital_kelly *= (1 + trade_kelly)
    capital_kelly_values.append(capital_kelly)
results = capital_kelly_values

# Convertir la colonne 'time' en format de date si nécessaire
data['time'] = pd.to_datetime(data['time'])

# Graphique de l'évolution du capital avec le levier spécifié
plt.figure(figsize=(14, 7))
plt.plot(data['time'], results, label=f'Levier {leverage}')
plt.xlabel('Date')
plt.ylabel('Capital')
plt.title('Évolution du capital avec levier pour US500Cash')
plt.legend()
plt.grid(True)
plt.show()
