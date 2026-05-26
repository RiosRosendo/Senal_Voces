# **🧠 Reconocimiento de Voz con Cuantización Vectorial**

## **📌 Propósito de la actividad**

El objetivo de esta práctica es comprender y aplicar las operaciones fundamentales involucradas en el reconocimiento de señales de voz mediante el uso de **cuantizadores vectoriales**.

Esta actividad forma parte de un desarrollo progresivo hacia la construcción de un sistema de reconocimiento de voz más robusto, con miras a su futura integración en un sistema robótico (por ejemplo, un *Puzzlebot* usando ROS2).

Se recomienda utilizar **Python**, ya que facilitará la integración futura con nodos *publisher* en ROS2.

---

## **🧩 Descripción general de la actividad**

En esta práctica se construirá un sistema básico de reconocimiento de voz siguiendo varias etapas del procesamiento digital de señales:

1. **Adquisición de datos (grabación de voz)**  
2. **Preprocesamiento de la señal**  
3. **Segmentación (detección de inicio y fin de palabra)**  
4. **Extracción de características (LPC)**  
5. **Entrenamiento con cuantización vectorial**  
6. **Evaluación del sistema (matriz de confusión)**

---

## **🔊 1\. Grabación de señales de voz**

Se deben grabar:

* 10 palabras diferentes (ejemplo: *start, stop, left, right, forward, backward, lift*, leave, break, continue)  
* Cada palabra debe repetirse **15 veces**  
* Frecuencia de muestreo: **16 kHz**

### **⚠️ Recomendaciones:**

* Grabar en un ambiente silencioso  
* Usar siempre la misma persona (consistencia en la voz)  
* Mantener condiciones similares en todas las grabaciones

---

## **🎚️ 2\. Preénfasis**

A cada señal de voz se le debe aplicar un filtro de preénfasis para resaltar las frecuencias altas:

Hp(z)=1−0.95z^−1

Este paso mejora la calidad del análisis posterior.

---

## **🪟 3\. Ventaneo (Hamming)**

Dividir la señal en bloques usando:

* Ventana de Hamming de **320 muestras**  
* Desplazamiento de **128 muestras** (overlap)

Esto permite analizar la señal en pequeñas secciones (frames).

---

## **✂️ 4\. Detección de inicio y fin de palabra**

Se debe identificar automáticamente dónde comienza y termina la palabra dentro de la señal.

### **Métodos sugeridos:**

* Energía de la señal  
* Cruces por cero (ZCR)  
* Umbrales adaptativos

Se proporciona un ejemplo en código (MATLAB) que puedes adaptar a Python.  
	

	% Cargar señal de voz

\[signal, fs\] \= audioread('fa.wav'); 

% Parámetros de ventana

frame\_length \= round(0.02 \* fs);  % 20 ms

hop\_length \= round(0.01 \* fs);    % 10 ms

num\_frames \= floor((length(signal) \- frame\_length)/hop\_length) ;

zcr \= zeros(1, num\_frames);

energy \= zeros(1, num\_frames);

% Calcular ZCR y energía

for i \= 1:num\_frames

    start\_idx \= (i-1)\*hop\_length \+ 1;

    frame \= signal(start\_idx : start\_idx \+ frame\_length \- 1);

    

    crossings \= sum(abs(diff(sign(frame)))) / 2;

    zcr(i) \= crossings / frame\_length;

    

    energy(i) \= sum(frame.^2) / frame\_length;

end

% Umbrales

zcr\_threshold \= 0.08 \* max(zcr);

energy\_threshold \= 0.03 \* max(energy);

% Detección de voz (inicio y fin)

voice\_flags \= (zcr \> zcr\_threshold) & (energy \> energy\_threshold);

% Buscar primer y último frame con voz

first\_voice\_frame \= find(voice\_flags, 1, 'first');

last\_voice\_frame \= find(voice\_flags, 1, 'last');

if isempty(first\_voice\_frame) || isempty(last\_voice\_frame)

    disp('No se detectó voz en la señal.');

    return;

end

% Convertir a índices de muestra

start\_sample \= (first\_voice\_frame \- 1)\*hop\_length \+ 1;

end\_sample \= (last\_voice\_frame \- 1)\*hop\_length \+ frame\_length;

% Asegurar que no salimos del límite

end\_sample \= min(end\_sample, length(signal));

% Recortar señal

signal\_trimmed \= signal(start\_sample:end\_sample);

% Reproducir y guardar (opcional)

sound(signal\_trimmed, fs);

audiowrite('voz\_recortada.wav', signal\_trimmed, fs);

% Visualización

t \= (0:length(signal)-1)/fs;

figure;

subplot(2,1,1);

plot(t, signal); hold on;

xline(start\_sample/fs, 'g--', 'Inicio detectado');

xline(end\_sample/fs, 'r--', 'Fin detectado');

title('Señal original con detección de inicio y fin de palabra');

subplot(2,1,2);

t\_trimmed \= (0:length(signal\_trimmed)-1)/fs;

plot(t\_trimmed, signal\_trimmed);

title('Señal recortada');

xlabel('Tiempo \[s\]');

### **Resultado esperado:**

* Señal recortada (solo la palabra)  
* Visualización del proceso

---

## **📊 5\. Extracción de características y cuantización vectorial**

### **🔹 Características:**

* Calcular coeficientes **LPC (Linear Predictive Coding)** de orden 12  
* Convertir LPC a **LSF (Line Spectral Frequencies)** para clustering

### **🔹 Cuantización vectorial:**

* Crear un **codebook** por cada palabra  
* Usar algoritmos de clustering (ej. K-means)

### **🔹 Distancia:**

* Comparar usando **distancia de Itakura-Saito**

### **❓ Pregunta clave:**

Evaluar el desempeño usando distintos tamaños de codebook:

* 16  
* 32  
* 64

Determinar cuál ofrece mejor rendimiento.

---

## **🧪 6\. Evaluación del sistema**

* Usar las **5 grabaciones restantes** de cada palabra (no usadas en entrenamiento)  
* Clasificar cada muestra usando los codebooks generados  
* Construir una **matriz de confusión**

### **📈 Objetivo:**

Evaluar qué tan bien el sistema reconoce cada palabra.

---

## **📦 Entregables esperados**

* Código en Python bien estructurado  
* Grabaciones organizadas por palabra  
* Codebooks generados  
* Matriz de confusión  
* Análisis de resultados (incluyendo comparación de tamaños de codebook)

---

## **💡 Notas finales**

* Mantener orden en archivos y nombres  
* Documentar cada paso del proceso  
* Validar resultados intermedios (no esperar hasta el final)  
* Pensar en escalabilidad hacia ROS2  
* Preguntar por archivos necesarios para completar la actividad  
* Breve resumen de lo que se debe de ver al correr cada codigo o que se debe hacer 

# 🧠 Extensión del Proyecto: Reconocimiento de Palabras Aisladas usando HMM + VQ

## 📌 Objetivo de esta extensión

Además de implementar cuantización vectorial, el sistema deberá evolucionar hacia un reconocedor probabilístico basado en:

- **MFCC (Mel Frequency Cepstral Coefficients)** para extracción de características
- **Vector Quantization (VQ)** para discretización
- **Hidden Markov Models (HMM)** para modelado temporal de palabras

El propósito es construir un sistema de reconocimiento de palabras aisladas más robusto y escalable, compatible posteriormente con sistemas robóticos en ROS2 y plataformas embebidas como Jetson Nano o Raspberry Pi.

---

# 🧩 Arquitectura General del Sistema

El flujo completo del sistema deberá seguir la siguiente secuencia:

```text
Audio → Preprocesamiento → Segmentación → MFCC
      → Cuantización Vectorial (VQ)
      → Secuencia Discreta de Índices
      → Modelo HMM
      → Clasificación
```

---

# 🎤 1. Representación de las Palabras

Cada grabación debe transformarse en una secuencia discreta de símbolos.

## Pipeline requerido

### Paso 1 — Extracción MFCC

De cada frame de voz extraer:

- MFCCs (13 coeficientes recomendados)
- Opcional:
  - Delta
  - Delta-Delta

---

### Paso 2 — Cuantización Vectorial

Todos los vectores MFCC deben ser agrupados usando clustering.

## Restricción obligatoria

El codebook global debe contener exactamente:

\[
256 \text{ vectores}
\]

Se recomienda usar:

- K-Means
- LBG (Linde-Buzo-Gray)

---

## Resultado esperado

Cada vector MFCC será reemplazado por el índice del centroide más cercano.

Por lo tanto, cada palabra quedará representada como:

```text
O = [12, 12, 15, 200, 201, 45, ..., 193]
```

donde cada número pertenece al rango:

```text
0 → 255
```

---

# 🏗️ 2. Modelado HMM

Se deben crear:

```text
10 modelos HMM
```

Uno por cada palabra del vocabulario.

Ejemplo:

- HMM_start
- HMM_stop
- HMM_left
- HMM_right
- etc.

---

# 🔄 3. Topología del Modelo

## Tipo obligatorio

Los HMM deben ser:

```text
Bakis / Left-to-Right
```

---

## Número de estados

Se recomienda:

```text
4 → 8 estados
```

dependiendo de la complejidad fonética de la palabra.

---

## Restricción de transición

La matriz de transición A debe permitir únicamente:

\[
a_{ii}
\]

(transición al mismo estado)

y

\[
a_{i,i+1}
\]

(transición al siguiente estado)

Todas las demás probabilidades deben ser:

\[
0
\]

---

## Ejemplo de matriz A

\[
A =
\begin{bmatrix}
0.7 & 0.3 & 0 & 0 & 0 \\
0 & 0.6 & 0.4 & 0 & 0 \\
0 & 0 & 0.8 & 0.2 & 0 \\
0 & 0 & 0 & 0.75 & 0.25 \\
0 & 0 & 0 & 0 & 1
\end{bmatrix}
\]

---

# 🛠️ 4. Entrenamiento mediante Ingeniería de Conteos

## ⚠️ Restricción importante

NO iniciar con:

- Baum-Welch aleatorio
- Librerías black-box
- HMMs preentrenados

---

## Método obligatorio inicial

Usar:

```text
Ingeniería de Conteos
```

---

# 📏 Segmentación Lineal

Cada secuencia observada debe dividirse uniformemente según el número de estados.

Ejemplo:

- Si el HMM tiene 5 estados:
  - Estado 1 → primeros 20%
  - Estado 2 → siguientes 20%
  - etc.

---

# 📊 Construcción de la Matriz B

La matriz B representa:

\[
P(observación \mid estado)
\]

---

## Procedimiento

Para cada estado:

1. Tomar los índices pertenecientes a su segmento
2. Contar frecuencia de aparición de cada símbolo
3. Normalizar probabilidades

---

## Dimensiones

Si existen:

- N estados
- M = 256 símbolos

Entonces:

\[
B \in \mathbb{R}^{N \times 256}
\]

---

# 🧪 Suavizado Obligatorio

Debido a que muchos símbolos podrían no aparecer durante entrenamiento:

Agregar:

\[
\epsilon = 10^{-6}
\]

a TODAS las entradas de B.

---

## Importante

Después del smoothing:

Cada fila debe cumplir:

\[
\sum_j B(i,j) = 1
\]

exactamente.

---

# 📈 Construcción de la Matriz A

Las probabilidades de transición deben estimarse usando:

- Duración promedio en cada estado
- Permanencia temporal del segmento

---

## Interpretación esperada

- Valores altos en diagonal:
  - permanencia en el estado
- Valores moderados en superdiagonal:
  - avance temporal

---

# ⚠️ 5. Advertencia sobre Baum-Welch

## Problema común

Inicializar Baum-Welch aleatoriamente suele producir:

- mínimos locales
- estados sin significado fonético
- colapso del modelo

---

## Uso correcto

Baum-Welch SOLO debe utilizarse:

```text
como refinamiento opcional
```

después de que el modelo basado en conteos funcione correctamente.

---

## Criterio práctico

Si al aplicar Baum-Welch:

- disminuye la precisión
- aumenta la confusión

entonces:

- la inicialización fue incorrecta
- faltan datos
- o el modelo aún no está estable

---

# 🧠 6. Reconocimiento usando Forward Algorithm

Para clasificar una palabra nueva:

1. Extraer MFCC
2. Aplicar VQ
3. Obtener secuencia discreta
4. Evaluar la secuencia contra los 10 HMMs

---

# 📉 Problema Numérico: Underflow

Como se multiplican muchas probabilidades pequeñas:

\[
P(O|\lambda)
\]

puede tender rápidamente a cero.

---

# ✅ Solución obligatoria

Implementar el algoritmo Forward en:

```text
espacio logarítmico
```

usando:

\[
\log(a \cdot b)
=
\log(a)+\log(b)
\]

---

# 🏆 Decisión Final

La palabra reconocida será:

```text
el modelo con mayor log-likelihood
```

---

# 📊 7. Validación del Sistema

El sistema debe demostrar que:

- realmente aprendió patrones fonéticos
- no es únicamente una caja negra

---

# 📌 Evidencias Obligatorias

## 1. Sparsity de B

Mostrar que:

- pocos índices tienen alta probabilidad
- la mayoría son cercanos a cero

---

## 2. Estructura de A

Mostrar que la matriz:

- tiene diagonal dominante
- respeta progresión temporal

---

## 3. Matriz de Confusión

Presentar matriz:

\[
10 \times 10
\]

Analizar:

- palabras confundidas
- similitud fonética
- errores de cuantización
- errores de segmentación

---

# 📦 Entregables Actualizados

El proyecto final deberá incluir:

- Grabaciones organizadas
- Código modular en Python
- Extracción MFCC
- Implementación VQ
- Codebook de 256 símbolos
- HMMs entrenados
- Algoritmo Forward logarítmico
- Matrices A y B
- Matriz de confusión
- Análisis de resultados
- Comparación entre:
  - LPC vs MFCC
  - Distintos tamaños de codebook
  - Con y sin Baum-Welch

---

# 🤖 Escalabilidad hacia ROS2

La implementación debe considerar futura integración con:

- ROS2 Nodes
- Publicadores de comandos de voz
- Puzzlebot
- Jetson Nano
- Tiempo real

Por lo tanto se recomienda:

- Código modular
- Separación entre entrenamiento e inferencia
- Pipeline reutilizable
- Serialización de modelos entrenados

---

# 💡 Recomendaciones Técnicas Finales

## Estructura sugerida de carpetas

```text
dataset/
    start/
    stop/
    left/
    right/

models/
    hmm_models/
    codebooks/

src/
    preprocessing/
    feature_extraction/
    vq/
    hmm/
    evaluation/
```

---

## Librerías recomendadas

Python:

- numpy
- scipy
- librosa
- sklearn
- matplotlib

Opcional:

- hmmlearn (solo referencia)
- numba
- joblib

---

# 🎯 Objetivo Final del Proyecto

Construir un sistema capaz de:

- reconocer palabras aisladas
- modelar dinámicas temporales
- generalizar nuevas muestras
- operar en sistemas robóticos reales

utilizando fundamentos clásicos de:

- DSP
- Machine Learning
- Reconocimiento Estadístico de Patrones
- Modelos Ocultos de Markov (HMM)