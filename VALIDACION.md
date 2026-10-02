# Validación del entregable

Se completaron dos ejecuciones desde kernel limpio en CPU. La segunda incorpora el diagnóstico del checkpoint final y amplía la referencia de copia a todas las variantes de entrenamiento.

- Notebook final: **39 celdas**, de las cuales **24 son código**, todas ejecutadas y sin salidas de error.
- Esquema Jupyter validado con `nbformat.validate`.
- Entrenamiento repetido desde cero: diferencia absoluta máxima **0.0** en NLL de train, validación y prueba, perplejidad e intervalos de los cuatro modelos principales.
- Las 40 generaciones de los experimentos principales coincidieron exactamente entre ejecuciones.
- El tokenizador BPE reconstruido conservó su hash.
- Se comprobó que ningún conjunto comparte familias con otro, que las variantes son únicas y que BPE reconstruye exactamente todos los textos.
- La prueba de máscara causal confirmó que cambiar el sufijo no altera las predicciones del prefijo.
- La generación con una misma semilla se repitió sin cambios.

El resumen legible por máquina está en `artifacts/verificacion.json`; configuración, versiones y hashes están en `artifacts/manifiesto.json`. El entorno de referencia fue Python 3.9.6, PyTorch 2.8.0 y CPU con cuatro hilos. El tiempo acumulado de entrenamiento de las tres redes fue aproximadamente 43 segundos en esta máquina, sin contar EDA, inferencia y arranque del kernel.

Estas comprobaciones no garantizan igualdad bit a bit entre arquitecturas de CPU, bibliotecas numéricas o versiones distintas. Tampoco prueban calidad literaria. La fluidez, la generalización y la copia se analizan por separado en el notebook.
