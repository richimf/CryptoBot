# Cripto Bot
## Experimentacion y Resultados
Se comienza a experimentar con los datos del historial de precio del Bitcoin BTC en dolares (de un archivo CSV). Estos datos servirán para entrenar al agente del algoritmo de **Q-learning**. La idea es poder invertir dinero en cierto tiempo y obtener pequeñas ganancias en intervalos de tiempo cortos.

Iniciamos el experimento con un saldo inicial o balance, una cantidad positiva fija de criptomonedas y un numero de episodios, los cuales están definidos por la *longitud de los datos del archivo CSV / el tamaño del episodio*. Teniendo estos datos, comenzamos a iterar en cada **step**, cada step es definido como el precio del bitcon en el tiempo **t**.

Dentro de nuestro código nos apoyamos de la función Sigmoide para obtener un arreglo de tres valores en el siguiente orden **[BUY, SELL,  HOLD]**, luego entonces, elegimos la acción con el mayor valor de probabilidad.
Cada acción actualiza el *balance*, el número de bitcoins adquiridos y la ganancia final. Recordemos que la idea es obtener la mayor cantidad de ganancia posible, para ello nos apoyamos en la siguiente ecuación, donde `p(t)` es el precio del bitcoin en el tiempo `t`.

> $reward = bitcoins*[p(t)/p(t-1)-1]$

En cada iteración actualizamos el **Q-table** con los valores del estado actual `s`, la acción elegida `a`, la ganancia obtenida o `reward r` y el valor del siguiente estado `s'`.
Una vez actualizado el *Q-table*, utilizamos los valores de `s, a, r, s'` para alimentar la formula de acción valor, obtener el siguiente estado y volver a iterar dentro del episodio.

[Ver Documentación completa](https://github.com/richimf/CryptoBot/blob/master/reporte/trading-bitcoin-reinforcement/main.pdf)


## Requirements
**Environment** de Anaconda con los siguientes paquetes:
- Keras
- Tensorflow
- Numpy
- Pandas
- Matplotlib

