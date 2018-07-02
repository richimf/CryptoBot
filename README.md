# Cripto Bot
## Experimentacion y Resultados
Se comienza a experimentar con los datos del historial de precio del Bitcoin BTC en dolares (de un archivo CSV). Estos datos servirán para entrenar al agente del algoritmo de Q-learning. La idea es poder invertir dinero en cierto tiempo y obtener pequeñas ganancias en intervalos de tiempo cortos.

Iniciamos el experimento con un saldo inicial o balance, una cantidad positiva fija de criptomonedas y un numero de episodios, los cuales están definidos por la *longitud de los datos del archivo CSV / el tamaño del episodio*. Teniendo estos datos, comenzamos a iterar en cada **step**, cada step es definido como el precio del bitcon en el tiempo **t**.

Dentro de nuestro código nos apoyamos de la función Sigmoide para obtener un arreglo de tres valores en el siguiente orden **[BUY, SELL,  HOLD]**, luego entonces, elegimos la acción con el mayor valor de probabilidad.
Cada acción actualiza el \textit{balance}, el número de bitcoins adquiridos y la ganancia final. Recordemos que la idea es obtener la mayor cantidad de ganancia posible, para ello nos apoyamos en la siguiente ecuación, donde $p(t)$ es el precio del bitcoin en el tiempo $t$.
\begin{center}
$reward = bitcoins*[p(t)/p(t-1)-1]$
\end{center}
En cada iteración actualizamos el Q-table con los valores del estado actual \textbf{s}, la acción elegida \textbf{a}, la ganancia obtenida o reward \textbf{r } y el valor del siguiente estado \textbf{s'}.
Una vez actualizado el Q-table, utilizamos los valores de \textbf{s, a, r, s'} para alimentar la formula de acción valor, obtener el siguiente estado y volver a iterar dentro del episodio.\\
\\
Los resultados  finales son variables, pero en todos los casos se obtiene una ganancia. En la siguiente gráfica se observa que iniciamos con un Balance cercano a \textit{200 USD}, con muchas alzas y bajas pero con tendencia positiva, al final la ganancia es cercana a los \textit{1000 USD}.




## Requirements
**Environment** de Anaconda con los siguientes paquetes:
- Keras
- Tensorflow
- Numpy
- Pandas
- Matplotlib

