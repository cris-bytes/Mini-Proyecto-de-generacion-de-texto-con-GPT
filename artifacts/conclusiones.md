**Resultados medidos**

- La selección por validación eligió **GRU-128**. Su perplejidad de prueba fue **89.61**
  frente a **168.97** del bigrama: una reducción relativa de **47.0%**.
  Su intervalo por familias fue **[77.92, 102.82]**; no incorpora variación entre entrenamientos.
- En la ablación de profundidad, **tres capas obtuvieron menor NLL de validación**. Tener más parámetros no garantiza generalizar mejor;
  esta comparación mantiene épocas y datos, no coste computacional.
- La recuperación obtuvo hit@1 temático **90%** en diez consultas manuales.
  La coincidencia de cinco palabras en la continuación pasó de **7.1%**
  sin recuperación a **0.0%** con recuperación. Son indicadores exploratorios,
  no una evaluación humana de relevancia o creatividad.
- El checkpoint final tiene una coincidencia media de cinco palabras de **94.1%**
  en sus generaciones, frente a **7.1%** del checkpoint temprano
  de la misma arquitectura. Se debe examinar si ganar fluidez significa reproducir las estructuras conocidas.
- El experimento aprendió patrones de un corpus pequeño, repetitivo y sintético. La separación por familias
  evita compartir la misma plantilla, pero los temas y el estilo fueron diseñados por una misma fuente.
  No es válido extrapolar estos resultados a cuentos de autores humanos ni a un producto editorial.

**Respuesta al problema:** la implementación funciona como laboratorio, pero las muestras del checkpoint seleccionado
presentan palabras deformadas, cambios de personaje y fragmentos sin conexión. **La utilidad como asistente literario
no queda demostrada**. Tener menor NLL que un bigrama y terminar con EOS no resuelve la coherencia narrativa.
El valor de recuperar una frase debe equilibrarse con su copia explícita y la posible continuación memorizada.

**Siguientes pasos:** reunir textos con permiso de uso de varios autores y separar por autor; repetir el entrenamiento
con al menos tres semillas; comparar BPE y un modelo preentrenado ajustado; hacer una evaluación ciega con lectores
que puntúen coherencia, sorpresa y utilidad; comparar recuperación frente a un prefijo humano de igual longitud.
