# Asistente de Yoga mediante Visión por Ordenador

## Descripción
Este proyecto desarrolla un sistema de reconocimiento de posturas de yoga utilizando visión computarizada. Se emplea una Raspberry Pi 4 y una cámara Raspberry (Module 3) para analizar y comparar posturas de yoga con un conjunto de datos predefinido. Se parte de una base de datos que incluye cinco posturas de yoga: el perro boca abajo, el árbol, la pluma de pavo real, el cuervo y el guerrero.

## Características del programa
1. **Configuración Inicial:** Se calibra la cámara con el patrón ChArUco.
2. **Sistema de Seguridad:** Para acceder, el usuario debe introducir una secuencia de figuras planas a modo de contraseña (estrella, cuadrado, pentágono). Estas han de ser introducidas al sistema de una en una.
3. **Reconocimiento de Posturas:** El sistema compara las posturas mostradas a la cámara con las almacenadas en la base de datos, utilizando algoritmos de comparación de puntos clave.

## Archivos del programa
- `Proyecto_AsesorYoga.py`: Script principal de Python que ejecuta el programa.
- `Proyecto_AsesorYoga.ipynb`: Jupyter Notebook con el código del programa.
- `requirements.txt`: Archivo que indica todas las bibliotecas y dependencias necesarias.
- `matrix.txt`: Archivo con la matriz de parámetros intrínsecos de la cámara.
- `distCoeffs.txt`: Archivo con los coeficientes de distorsión de la cámara.
- `charuco.png`: Patrón de calibración ChArUco.
- `imagenes_calibracion`: Directorio con las imágenes empleadas para calibrar la cámara.
- `Info`: Directorio con los archivos de texto con la información sobre las posturas.
- `plantillas`: Directorio con las imágenes de posturas de yoga que forman la base de datos.
- `resultados_calibracion`: Directorio con dos subcarpetas, con las imágenes mostrando los marcadores ArUco y los Chessboard corners identificados.

## Ejecución del programa
1. Instalar las dependencias necesarias (detalladas en el archivo `requirements.txt`).
2. Descargar los archivos del repositorio (`Proyecto_AsesorYoga.py`, `matrix.txt`, `distCoeffs.txt`, `imagenes_calibracion`, `info`, `plantillas`) y colocarlos en el mismo directorio que el programa.
3. Ejecutar el script principal `ProyectoFinal_ElenayVictoria.py` desde la Raspberry Pi con la cámara conectada.
4. Seguir las instrucciones en pantalla para la calibración, el sistema de seguridad y el reconocimiento de posturas.
