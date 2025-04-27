# 🛫 Análisis Predictivo de Retrasos Aéreos (2019)

## 📌 Contexto
Proyecto de ciencia de datos para predecir retrasos en vuelos comerciales, con el objetivo de optimizar la gestión operativa de aerolíneas. Desarrollado como parte de [nombre del programa o curso].

## 🗃️ Dataset
- **Fuente**: [Kaggle - 2019 Airline Delays and Cancellations](https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations)
- **Registros**: 6.5+ millones de vuelos
- **Variables clave**:
  - Operacionales: aerolínea, aeropuerto, hora de vuelo
  - Climáticas: precipitación (PRCP), viento (AWND), nieve (SNOW)
  - Técnicas: antigüedad del avión, asientos
  - Target: `DEP_DEL15` (retraso >15 min)

## 🔍 Hipótesis Validadas
1. **Climática**: Eventos adversos aumentan retrasos (p<0.001)
2. **Antigüedad**: Aviones 11-15 años tienen mayor riesgo (19.9%)
3. **Temporal**: Picos en diciembre (23.5%) y julio (21.8%)

## 🤖 Modelos Implementados
| Modelo          | Recall | Precisión | AUC  |
|-----------------|--------|-----------|------|
| Random Forest   | 66.0%  | 28.8%     | 0.64 |
| XGBoost         | 8.2%   | 68.6%     | 0.54 |
| Regresión Log.  | 1.3%   | 52.8%     | 0.51 |

## 📊 Resultados Clave
- **Mejor modelo**: Random Forest (mejor balance recall/precisión)
- **Desafío principal**: Desbalance de clases (81% no retrasos)
- **Hallazgo crítico**: Variables climáticas son predictores clave
