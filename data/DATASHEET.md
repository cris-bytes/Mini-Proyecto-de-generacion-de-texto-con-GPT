# Ficha de datos: Microcuentos controlados

## Procedencia

Corpus nuevo preparado para este ejercicio con asistencia de IA. No se extrajo de WikiHow, de los notebooks guía ni de una colección literaria externa. `familias.json` conserva las 60 plantillas fuente. No se atribuye la redacción a autores humanos ni se presenta como observación de comportamiento real.

## Construcción

Cinco temas: memoria, ciudad, naturaleza, tecnología y mar. Cada tema tiene 12 familias. Las plantillas sustituyen tres campos: nombre de personaje, lugar y momento. La función `build_corpus` genera 16 variantes únicas por familia, para un total de 960 registros. Los personajes son ficticios.

El balance es deliberado. La expansión aumenta observaciones pero **no aporta 16 estructuras narrativas nuevas**. Las variantes comparten prácticamente toda la redacción. Además, diferentes familias comparten motivos y expresiones, como cartas, ventanas, espera y recuerdos.

## Esquema

| Campo | Significado |
|---|---|
| `id` | Identificador tema-familia-variante |
| `familia` | Unidad narrativa independiente para partición y bootstrap |
| `tema` | Condición de generación definida durante la creación |
| `split` | train, val o test |
| `texto` | Texto con sus tres campos sustituidos |
| `n_palabras` | Conteo de secuencias alfabéticas, incluidas tildes |

## Particiones

Semilla 17. Por tema, 8 familias van a train, 2 a val y 2 a test. Resultado: 640/160/160 filas y 40/10/10 familias. Se decide la partición antes de expandir los textos. BPE y BM25 usan únicamente train. Test no selecciona checkpoints.

La separación evita compartir una misma plantilla, pero no elimina similitudes semánticas ni estilo común entre conjuntos. Para nuevos dominios harían falta documentos independientes y particiones por fuente o autor.

## Calidad y limitaciones

- Se verifican unicidad exacta, ausencia de nulos, separación de familias y representación exacta por BPE.
- No hay anotación humana independiente de calidad literaria.
- Algunos reemplazos de nombre no modifican el género gramatical de pronombres del texto; este sesgo forma parte de los límites del corpus.
- Los temas son etiquetas de diseño, no juicios consensuados por anotadores.
- La ficción incluye pérdida, despedida y memoria. No ofrece consejos ni información factual.
- Las métricas sobre diez familias de prueba tienen alcance limitado. No usar estos resultados para afirmar competencia general en escritura.

## Uso previsto

Experimentación docente y reproducible con modelos causales pequeños y recuperación. El material fue creado como parte de este entregable; se permite su reutilización educativa con reconocimiento de su procedencia sintética y asistida. No se incorpora texto de terceros sujeto a una licencia externa de corpus.
