# Proyecto-Final Modulo The Bridge
# Autores
Lead Instructor: Iraitz Montalbán

Teacher Assistant: Luis Miguel Andújar Baena

Alumno: Mikel Cerio Chinchurreta
# Link dataset
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
# Descripción del Conjunto de Datos

Este proyecto utiliza un conjunto de datos para predecir si un usuario realizará una compra en un e-commerce. A continuación, se describen las variables incluidas en el conjunto de datos.

El conjunto de datos consta de vectores de características que pertenecen a 12.330 sesiones. 
El conjunto de datos se formó de manera que cada sesión
pertenecería a un usuario diferente en un período de 1 año para evitar
cualquier tendencia a una campaña específica, día especial, usuario
perfil, o punto. 


## Variables de Características

- **Administrative**
  - **Descripción:** Número de páginas vistas por el usuario en la sección administrativa del sitio web.
  - **Tipo:** Numérico
  - **Ejemplo de Valor:** 3

- **Administrative_Duration**
  - **Descripción:** Tiempo total (en segundos) que el usuario ha pasado en la sección administrativa.
  - **Tipo:** Numérico
  - **Ejemplo de Valor:** 120.5

- **Informational**
  - **Descripción:** Número de páginas vistas por el usuario en la sección de información del sitio web.
  - **Tipo:** Numérico
  - **Ejemplo de Valor:** 2

- **Informational_Duration**
  - **Descripción:** Tiempo total (en segundos) que el usuario ha pasado en la sección informativa.
  - **Tipo:** Numérico
  - **Ejemplo de Valor:** 45.0

- **ProductRelated**
  - **Descripción:** Número de páginas vistas por el usuario relacionadas con productos.
  - **Tipo:** Numérico
  - **Ejemplo de Valor:** 15

- **ProductRelated_Duration**
  - **Descripción:** Tiempo total (en segundos) que el usuario ha pasado en páginas relacionadas con productos.
  - **Tipo:** Numérico
  - **Ejemplo de Valor:** 300.0

- **BounceRates**
  - **Descripción:** Tasa de rebote, que representa el porcentaje de visitas en las que el usuario solo ve una página y luego abandona el sitio.
  - **Tipo:** Numérico (0 a 1)
  - **Ejemplo de Valor:** 0.2

- **ExitRates**
  - **Descripción:** Tasa de salida, que representa el porcentaje de usuarios que abandonan el sitio desde una página específica.
  - **Tipo:** Numérico (0 a 1)
  - **Ejemplo de Valor:** 0.15

- **PageValues**
  - **Descripción:** Valor asignado a las páginas vistas por el usuario, que puede reflejar la importancia o relevancia de la página para la conversión.
  - **Tipo:** Numérico
  - **Ejemplo de Valor:** 50.0

## Variable Objetivo

- **Revenue**
  - **Descripción:** Indicador binario que señala si el usuario realizó una compra o no.
  - **Tipo:** Binario (0 = No, 1 = Sí)
  - **Ejemplo de Valor:** 1

