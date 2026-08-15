# Sistema de Recomendación de Películas (TFG)

*[Read in English](README.en.md)*

Trabajo de Fin de Grado en el que se construye y compara un sistema de recomendación de películas sobre el dataset **MovieLens 100K**, enriquecido con información de contenido obtenida de la API de **TMDB**. Se implementan y evalúan tres familias de enfoques: filtrado basado en contenido, filtrado colaborativo y modelos híbridos.

> 📄 **Memoria completa del proyecto:** [documentos/Memoria_proyecto.pdf](documentos/Memoria_proyecto.pdf)

![Esquema del pipeline del proyecto: Importation → Cleaning → EDA → Transform → Model, apoyado en los datos Raw → Processed → Model ready](documentos/esquema_proyecto.png)

## Objetivo del proyecto

El proyecto consiste en probar, comparar y evaluar distintos algoritmos y métodos de filtrado para generar recomendaciones de películas, entrenándolos y evaluándolos sobre el mismo conjunto de datos y las mismas particiones de train/val/test para poder comparar sus resultados de forma consistente. Concretamente se exploran:

- **Filtrado basado en contenido**: recomienda películas similares a partir de su información textual (sinopsis, géneros, tags), mediante TF-IDF y embeddings de SBERT.
- **Filtrado colaborativo**: recomienda a partir del histórico de valoraciones de los usuarios, sin usar información de contenido, mediante KNN (basado en memoria), SVD (factorización de matrices) y redes neuronales de filtrado colaborativo (NCF).
- **Modelos híbridos**: combinan las señales de contenido y colaborativas (por regresión, fusión de rankings o redes neuronales) para intentar mejorar el rendimiento de cada enfoque por separado.

Los resultados de cada algoritmo se recogen y comparan en la sección [Resultados](#resultados-rmse-sobre-test).

## Fuentes de datos

- **MovieLens 100K** (https://grouplens.org/datasets/movielens/): ratings, tags, movies y links, con ~100.000 valoraciones de 610 usuarios sobre unas 9.700 películas.
- **TMDB** (https://developer.themoviedb.org/reference/intro/getting-started): overview, tagline, géneros, popularidad, presupuesto, recaudación, duración, etc., obtenidos mediante llamadas a su API a partir del `tmdbId` de cada película (vía `links.csv`).

## Estructura del proyecto

El proyecto sigue un pipeline secuencial de cuadernos Jupyter, numerados por etapa:

```
01_importation/   Descarga de los datos crudos (MovieLens + API de TMDB)
02_cleaning/      Limpieza, validación e integridad referencial de cada dataset
03_EDA/           Análisis exploratorio de los datos ya validados
04_transform/     Construcción de los datasets listos para entrenar los modelos
05_model/         Entrenamiento y evaluación de los modelos de recomendación
data/             Datos crudos, procesados y listos para modelar (no versionados)
documentos/       Memoria del TFG y esquema del proyecto
logs/             Logs de ejecución (p. ej. errores de la API de TMDB)
```

### 01_importation

- **importation.ipynb**: descarga los datos de MovieLens (versión 100K) y, para cada película, llama a la API de TMDB usando el `tmdbId` de `links.csv`. Cada respuesta se guarda como un JSON individual en `data/01_raw/TMDB`. Las peticiones fallidas quedan registradas en `logs/requestTMDB.log`. Requiere un token de TMDB en la variable de entorno `TMDB_TOKEN` (ver [Configuración](#configuración)).

### 02_cleaning

Un cuaderno de limpieza y validación por cada dataset de MovieLens, más uno final que cruza todas las fuentes:

| Cuaderno | Dataset | Qué hace |
|---|---|---|
| `clean_movies.ipynb` | movies.csv | Valida movieId y géneros, separa el año del título, marca sin género como nulo |
| `clean_ratings.ipynb` | ratings.csv | Valida unicidad, rango de fechas y escala de rating; convierte el timestamp a fecha |
| `clean_tags.ipynb` | tags.csv | Valida nulos y fechas; normaliza el texto de las etiquetas |
| `clean_links.ipynb` | links.csv | Valida movieId/tmdbId; cruza con el log de errores de la API; elimina imdbId |
| `clean_TMDB.ipynb` | JSON de TMDB | Consolida los JSON individuales en un único dataset con las columnas de interés |
| `intengridadRefencial.ipynb` | los 5 anteriores | Cruza todos los datasets ya limpios y elimina los registros huérfanos (películas sin TMDB, sin género en ningún sistema, etc.), garantizando que todos comparten el mismo conjunto de `movieId`/`userId` |

Salida: `data/02_processed/*_clean.parquet` y `data/02_processed/*_integrity.parquet`.

### 03_EDA

- **eda.ipynb**: análisis exploratorio sobre los datos ya validados: distribución de ratings por usuario y por película, evolución temporal, distribución de géneros (TMDB vs. MovieLens), y análisis de la dispersión (sparsity) de la matriz usuario-película (~98.3%), que condiciona el diseño de los modelos colaborativos.

### 04_transform

Construye los datasets finales que consumen los modelos:

- **PLN.ipynb**: agrega los tags por película, combina título, géneros, tags y overview de TMDB en un único texto por película → `data/03_model_ready/NLP.parquet` (usado por los modelos de contenido TF-IDF y SBERT).
- **ratings.ipynb**: particiona los ratings en train/val/test, de forma temporal y por usuario (70/15/15) → `data/03_model_ready/ratings_{train,val,test}.parquet` (usado por todos los modelos colaborativos e híbridos).
- **hybrid.ipynb**: combina variables de TMDB (popularity, vote_average, vote_count, runtime) y de MovieLens (year, genres codificados en multi-hot) con los ratings → `data/03_model_ready/hybrid.parquet` (usado por los modelos híbridos basados en red neuronal).

### 05_model

Los modelos se organizan en tres familias, cada una en su propia subcarpeta:

**01_content_based** — filtrado basado en el contenido de la película (overview, tags, géneros):
- `TFIDF.ipynb`: vectorización TF-IDF (con preprocesamiento spaCy: lematización y stopwords) + similitud del coseno.
- `SBERT.ipynb`: embeddings de sentence-transformers (`all-MiniLM-L6-v2`) + similitud del coseno.

**02_collaborative** — filtrado colaborativo, a partir únicamente de los ratings:
- `KNNitem.ipynb` / `KNNuser.ipynb`: KNN basado en memoria (KNNBasic, KNNWithMeans, KNNBaseline de `surprise`), en variante item-based y user-based.
- `SVD.ipynb`: factorización de matrices con y sin términos de sesgo, comparado con un modelo baseline.
- `RNcol.ipynb` / `RNcolADAM.ipynb`: filtrado colaborativo neuronal (NCF) con embeddings de usuario/película, entrenado con SGD y con Adam respectivamente.

**03_hybrid** — combinación de las dos familias anteriores:
- `hybrid.ipynb`: combina SVD y SBERT, primero por regresión lineal y después fusionando sus rankings con Reciprocal Rank Fusion (RRF), comparado con popularidad y recomendaciones aleatorias.
- `RNhybrid.ipynb`: red NCF que añade a los embeddings de usuario/película las variables de contenido de TMDB, con análisis de interpretabilidad (SHAP) y del error según la popularidad de la película.
- `RNhybridMenosVar.ipynb`: misma red, con menos variables de contenido, para valorar su aportación real.

#### Resultados (RMSE sobre test)

| Modelo | RMSE |
|---|---|
| RNhybrid (NCF + variables TMDB, Adam) | **≈0.847** |
| SVD con sesgos | ≈0.861 |
| RNhybridMenosVar (NCF + menos variables) | ≈0.870 |
| BaselineOnly (solo sesgos usuario/película) | ≈0.886 |
| RNcolADAM (NCF puro, Adam) | ≈0.890 |
| KNNBaseline item-based | ≈0.883 |
| KNNBaseline user-based | ≈0.906 |
| SVD sin sesgos | ≈0.919 |
| RNcol (NCF puro, SGD) | ≈1.058 |

Los modelos de contenido (TF-IDF, SBERT) y el híbrido SVD+SBERT se evalúan aparte con precisión/recall@10, al no predecir directamente un rating; el detalle de todas las métricas está en cada cuaderno y en la memoria.

## Configuración

Requiere Python 3.11 y [uv](https://docs.astral.sh/uv/) como gestor de dependencias. Las dependencias del proyecto están definidas en [pyproject.toml](pyproject.toml) (versiones exactas fijadas en [uv.lock](uv.lock)), ambos en la raíz del repositorio.

```bash
uv sync
```

Para poder ejecutar `01_importation/importation.ipynb` hace falta un archivo `.env` en la raíz del proyecto con un token de TMDB:

```
TMDB_TOKEN=tu_token_de_tmdb
```

## Datos

La carpeta `data/` no está versionada en git (solo su estructura de carpetas, mediante archivos `.gitkeep`): hay que generarla ejecutando el pipeline desde `01_importation` en adelante, o copiar los archivos de datos manualmente en:

```
data/01_raw/movieLens/    (movies.csv, ratings.csv, tags.csv, links.csv)
data/01_raw/TMDB/         (un .json por película)
data/02_processed/        (generado por 02_cleaning)
data/03_model_ready/      (generado por 04_transform)
```
