# 🛫 Análisis Predictivo de Retrasos Aéreos (2019)

## 📌 Contexto
Proyecto de ciencia de datos para predecir retrasos en vuelos comerciales, con el objetivo de optimizar la gestión operativa de aerolíneas. Incluye modelado predictivo y visualización interactiva.

## 🌟 Dashboard Interactivo
Explorar las relaciones clave entre variables en **[Dashboard PowerBI](https://nachols1986.github.io/infovis/airport_delays.html)**  
*(Visualización dinámica de patrones de retrasos, clima y operaciones)*

## 🗃️ Dataset
- **Fuente**: [Kaggle - 2019 Airline Delays and Cancellations](https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations)
- **Registros**: 6.5+ millones de vuelos
- **Variables clave**:
  - Operacionales: aerolínea, aeropuerto, hora de vuelo
  - Climáticas: precipitación (PRCP), viento (AWND), nieve (SNOW)
  - Técnicas: antigüedad del avión, asientos
  - Target: `DEP_DEL15` (retraso >15 min)

## 🔍 Hallazgos Principales
1. **Climáticos**: 
   - Vientos >15mph aumentan retrasos en 42% (OR=1.42)
   - Nieve cuadruplica la tasa base (de 18.9% a 60.2%)

2. **Temporales**:
   - Diciembre: +4.6pts sobre la media (23.5% retrasos)
   - Turno tarde-noche: 22% mayor riesgo que mañana

3. **Flota**:
   - Aviones 11-15 años: peak de retrasos (19.9%)

## 🤖 Modelado Predictivo
| Modelo          | Recall | Precisión | AUC  | Tiempo |
|-----------------|--------|-----------|------|--------|
| Random Forest   | 66.0%  | 28.8%     | 0.64 | 586s   |
| XGBoost         | 8.2%   | 68.6%     | 0.54 | 11s    |
| Regresión Log.  | 1.3%   | 52.8%     | 0.51 | 134s   |

**Insight**: Random Forest detecta 3/4 retrasos, pero con muchos falsos positivos.
