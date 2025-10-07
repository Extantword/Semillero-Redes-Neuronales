# Sesión 5

Ya hemos visto que el modelo de regresión lineal permite en principio aproximar cualquier combinación lineal de funciones, no solo líneas rectas. El problema radica en que para la gran mayoría de problemas prácticos, no conocemos cual es el proceso generador de datos, y por lo tanto no podemos simplemente "meterle" la función correcta al modelo de regresión lineal. Sin embargo, podríamos razonar que aunque si bien no conocemos exactamente cual es el proceso generador de datos, podemos intentar volver al modelo de regresión lineal extremadamente flexible métiendole todas las funciones que se nos ocurran, y que si lo volvemos lo suficientemente flexible, el modelo tendrá la capacidad necesaria para aproximar la función desconocida. Veamos que tan bien funciona esta idea. 

## Ajuste polinómico

¿Cuales funciones deberíamos empezar a meterle al modelo para que tenga la flexibilidad deseada? Una primera idea es usar polinomios: son funciones que entendemos bien, fáciles de evaluar y derivar, y tenemos teoremas matemáticos que nos dicen que en principio cualquier función continua puede ser aproximada tan bien como se desee con un polinomio. Teniendo lo anterior en cuenta, consideremos una función polinómica de la forma

$$\hat{y}(x, \textbf{w}) = w_0 + w_1x + w_2x^2 + ... + w_Mx^M = \sum_{j = 0}^M w_jx^j.$$

Los coeficientes del polinomio $w_0, ..., w_M$ están colectivamente denotados por el vector $\textbf{w}$. Notemos que aunque $y$ es una función no lineal de $x$, _si_ es una función lineal de los coeficientes $\textbf{w}$. Consideremos un conjunto de datos de la forma $X = [x_1 \quad \dots \quad x_n]^T$ con etiquetas $y = [y_1 \quad \dots \quad y_n]^T$, como se ve en la siguiente gráfica

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

Sin embargo, la curva oscila salvajemente, y no parece ser realmente una buen ajuste para el conjunto de datos, aunque el error cuadrático medio sea $0$. Recordemos que una de las cosas que queremos es que el modelo _generalice_, es decir, realice buenas predicciones sobre datos que nunca ha visto antes. Lo que está sucediendo es que le hemos concedido _demasiada_ flexibilidad al modelo, tanta que ha acabado por simplemente 

