# Sesión 5 Overfitting, regularización, y el sacrificio sesgo-varianza

Ya hemos visto que el modelo de regresión lineal permite en principio aproximar cualquier combinación lineal de funciones, no solo líneas rectas. El problema radica en que para la gran mayoría de problemas prácticos, no conocemos cual es el proceso generador de datos, y por lo tanto no podemos simplemente "meterle" la función correcta al modelo de regresión lineal. Sin embargo, podríamos razonar que aunque si bien no conocemos exactamente cual es el proceso generador de datos, podemos intentar volver al modelo de regresión lineal extremadamente flexible métiendole todas las funciones que se nos ocurran, y que si lo volvemos lo suficientemente flexible, el modelo tendrá la capacidad necesaria para aproximar la función desconocida. Veamos que tan bien funciona esta idea. 

## Ajuste polinómico

¿Cuales funciones deberíamos empezar a meterle al modelo para que tenga la flexibilidad deseada? Una primera idea es usar polinomios: son funciones que entendemos bien, fáciles de evaluar y derivar, y tenemos teoremas matemáticos que nos dicen que en principio cualquier función continua puede ser aproximada tan bien como se desee con un polinomio. Teniendo lo anterior en cuenta, consideremos una función polinómica de la forma

$$\hat{y}(x, \textbf{w}) = w_0 + w_1x + w_2x^2 + ... + w_Mx^M = \sum_{j = 0}^M w_jx^j.$$

Los coeficientes del polinomio $w_0, ..., w_M$ están colectivamente denotados por el vector $\textbf{w}$. Notemos que aunque $y$ es una función no lineal de $x$, _si_ es una función lineal de los coeficientes $\textbf{w}$. Consideremos un conjunto de datos de la forma $X = [x_1 \quad \dots \quad x_n]^T$ con etiquetas $y = [y_1 \quad \dots \quad y_n]^T$, generado aleatoriamente a partir de la función $sin(x)$, como se ve en la siguiente gráfica. 

[IMAGEN]

Como vimos en la sesión anterior, podemos encontrar el polinomio que mejor se ajusta a los datos minimizando el error cuadrático medio

$$MSE = \dfrac{1}{n}\sum_{n = 1}^N(\hat{y}(x_n, w) - y_n)^2,$$

y que de hecho existe una solución para el mejor $\textbf{w}^{*}$ que minimiza el error cuadrático medio

$$ \textbf{w}^{*} = (\tilde{X}\tilde{X}^T)^{-1}\tilde{X}y.$$

donde $\tilde{X}$ es la matriz de diseño aumentada con las nuevas características

$$\tilde{X} =
\begin{bmatrix}
    x_{1}       & x_{1}^2 & x_{1}^3 & \dots & x_{1}^M \\
    x_{2}       & x_{2}^2 & x_{3}^3 & \dots & x_{2}^M \\
   && \vdots\\
    x_{3}       & x_{3}^2 & x_{3}^3 & \dots & x_{N}^M
\end{bmatrix}$$

Lo único que faltaría es elegir cual va a ser el grado del polinomio, es decir, escoger $M$. Recordemos por la sesión anterior que esto es lo mismo que escoger cuantas características de la forma $x_n^i$ vamos a añadir a la matriz de diseño. Como el grado $M$ del polinomio es en principio algo que _nosotros_ escogemos, lo llamamos un **hiperparámetro**, a diferencia de los parámetros $w_0, ..., w_M$ que son _aprendidos_ por el modelo. Escojamos distintos valores para el hiperparámetro $M$ y veamos que sucede. Para $M = 1$ tenemos la siguiente curva

[IMAGEN M = 1]

Notemos que como es un polinomio de grado 1, corresponde a una recta, y por lo tanto no cuenta con la flexibilidad necesaria para ajustarse al conjunto de datos. Ahora, para $M = 3$ tenemos la siguiente curva

[IMAGEN M = 3]

parece que este polinomio se adecua bastante bien al conjunto de datos, y es una buena candidata para ser un modelo de regresión. Sin embargo, podemos seguir aumentando la flexibilidad del modelo. Para $M = 9$, tenemos la siguiente curva

[IMAGEN M = 9]

Notemos que para la curva anterior, tenemos un ajuste _perfecto_ para el conjunto de datos: el polinomio pasa por todos los puntos. Por lo tanto, el error cuadrático medio es cero

$$MSE = \dfrac{1}{n}\sum_{n = 1}^N(\hat{y}(x_n, w) - y_n)^2 = 0.$$

Sin embargo, la curva oscila salvajemente, y no parece ser realmente una buen ajuste para el conjunto de datos, aunque el error cuadrático medio sea $0$. Recordemos que una de las cosas que queremos es que el modelo _generalice_, es decir, realice buenas predicciones sobre datos que nunca ha visto antes. Viendo los coeficientes obtenidos para el polinomio, podemos obtener algo de claridad sobre lo que está pasando.

Notamos que los coeficientes son extremadamente grandes. Lo que está sucediendo es que le hemos concedido _demasiada_ flexibilidad al modelo, tanta que el modelo a acabado por simplemente encontrar el polinomio que se ajusta perfectamente al conjunto de datos, pero que no corresponde para nada al verdadero proceso generador de datos. 

El fénomeno anterior se conoce como **overfitting** o sobreajuste, y es extremadamente común encontrarse con este problema a la hora de entrenar un modelo. La moraleja que sacamos de todo esto es que darle flexibilidad al modelo no necesariamente se traduce en mejores resultados. 

Algo interesante es considerar que pasa cuando dejamos el grado del polinomio en $M = 9$, pero aumentamos considerablemente el conjunto de datos, como se ve en la siguiente gráfica

Notemos que el problema de overfitting se desvanece a medida que aumentamos la cantidad de datos. Otra forma de decir esto es que entre más grande sea el conjunto de datos, podremos permitirnos tener modelos más flexibles. 


## Regularización

Una técnica comunmente utilizada para controlar el problema de overfitting es llamada **regularización**, la cual consiste en introducir un término de penalización al error cuadrático medio, y obtener el _error cuadrático medio regularizado_

$$MSE_{\lambda} = \dfrac{1}{n}\sum_{n = 1}^N(\hat{y}(x_n, \textbf{w})  - y_n)+ \lambda||\textbf{w}||^2,$$

donde $||\textbf{w}||^2 = \textbf{w}^T\textbf{w} = w_0^2 + w_1^2 + ... + w_M^2$, y el coeficiente $\lambda$ controla la importancia del término de regularización con respecto al término del error cuadrático medio. Es decir, el término $||\textbf{w}||^2$ castiga elecciones grandes para los parámetros, y por lo tanto hará que a la hora de minimizar el polinomi, se prefieran polinomios con coeficientes pequeños, que no presenten oscilaciones tan grandes. Como $\lambda$ es escogido por nosotros, también es un **hiperparámetro**. En la siguiente gráfica vemos que pasa para una elección de $\ln(\lambda) = -18$

[LAMBDA LN(18)]

Vemos que al introducir la regularización, el problema de overfitting a casi que desaparecido. Sin embargo, para un $\lambda$ demasiado grande, digamos $\ln(\lambda) = 0$, el problema persiste. 


### Sesgo y Varianza

Volvamos al conjunto de datos anterior, generado con la función $sen(x)$. 


[IMAGEN]

Por un lado, si escogemos un grado muy pequeño para el polínomio, digamos $M = 1$, entonces el modelo corresponde a una línea recta y por lo tanto no tiene la flexiblidad para ajustarse al conjunto de datos. Vamos a llamar a este un problema de **sesgo**: estamos sesgando el modelo hacia un tipo de tipo de ajuste muy específico: líneas rectas. Otra forma de decirlo es que al poner el grado máximo del polinomio en $1$, estamos inculcándole un desagrado "infinito" hacia ajustes polinómicos de mayor grado. 


Por otro lado, si $M = 9$, le hemos entregado tanta flexibilidad al modelo que ahora puede simplemente _memorizarse_ el conjunto de datos sin tener que preocuparse por generalizar. Por lo tanto, si escojemos un nuevo conjunto de datos de entrenamiento, encontraremos que los resultados del modelo cambian y varían, ya que estaba memorizándose el conjunto de datos anterior. Llamemos a este un problema de **varianza**: estamos encontrando variaciones muy grandes de un conjunto de entrenamiento a otro. 

Vemos entonces que tanto el _sesgo_ como la _varianza_ están en conflicto el uno con el otro: cuando tenemos mucho sesgo, tenemos poca varianza, y cuando tenemos poca varianza, tenemos mucho sesgo. 

Este "conflicto" entre ambos conceptos es lo que llamamos el  **trade-off** o **sacrificio** sesgo-varianza. El trade-off es un problema central en el aprendizaje supervisado. Idealmente, uno quiere escoger un modelo que capture las regularidades en el conjunto de datos de entrenamiento, pero también que generalice a datos que nunca ha visto antes. Desafortunadamente, típicamente es dificil hacer los dos de manera simultánea. Métodos de alta varianza pueden representar bien el conjunto de entrenamiento pero corren el riesgo de tener sobreajustarse a datos ruidosos o pocos representativos. En contraste, métodos con alto sesgo tipicamente producen modelos más simples que fallan en capturar regularidades importantes en el conjunto de datos. 

Una forma de pensar en el trade-off de manera intuitiva y visual es la siguiente

[IMAGEN 1]
[IMAGEN 2]



Cada punto representa el resultado de haber entrenado el modelo en un conjunto de datos distinto. Para el primer caso de la esquina superior izquierda, vemos alto sesgo y poca varianza: el performance modelo no varía mucho entre conjuntos de datos, pero tiene un rendimiento pobre. Intutivamente podríamos hacer una "traslación" del modelo para mejorarlo.

 En la esquina inferior derecha vemos el caso de alta varianza y poco sesgo: el modelo varía mucho de un conjunto de entrenamiento a otro, pero los resultados están "al rededor" del que queremos. Intuitivamente, podríamos un sacar un promedio de todos los resultados para obtener uno mejor que todos los demás. Intente dibujar el caso de bajo varianza y bajo sesgo, y el caso de alta varianza y alto sesgo. 

