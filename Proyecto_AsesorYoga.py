# Importamos las librerias necesarias
import numpy as np
import cv2
import cv2.aruco as aruco
import pathlib
import time
from picamera2 import Picamera2
import glob

# Inicializamos el sift y el bf que vamos a usar para el matching
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

def calibrar_charuco(ruta_directorio, formato_imagen, tamano_marker, tamano_casilla):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    # 5 cuadrados en X y 7 cuadrados en Y
    board = aruco.CharucoBoard_create(5, 7, tamano_casilla, tamano_marker, aruco_dict)
    parametros_aruco = aruco.DetectorParameters_create()

    corners_list, id_list = [], []
    img_dir = pathlib.Path(ruta_directorio)
    
    # Buscamos los marcadores ArUco y los Chessboard corners en cada imagen
    for img in img_dir.glob(f'*.{formato_imagen}'):
        print(f'using image {img}')
        image = cv2.imread(str(img))
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        corners, ids, rejected = aruco.detectMarkers(
            img_gray, 
            aruco_dict, 
            parameters=parametros_aruco
        )

        resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=img_gray,
            board=board)
        
        # Si se ha encontrado un tablero Charuco, guardamos los puntos de la imagen (corners)
        # Establecemos un umbral de 20 cuadrados
        if resp > 20:
            corners_list.append(charuco_corners)
            id_list.append(charuco_ids)
        
        # Para cada imagen, dibujamos los marcadores ArUco y los Chessboard corners
        image_aruco = aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imwrite(f'resultados_calibracion/aruco/aruco_{img.parts[-1]}', image_aruco)

        image = cv2.imread(str(img))
        image_chessboard = aruco.drawDetectedCornersCharuco(image, charuco_corners)
        cv2.imwrite(f'resultados_calibracion/chessboard//chessboard_{img.parts[-1]}', image_chessboard)

    # Calibramos la camara con los puntos obtenidos
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_list,
        charucoIds=id_list,
        board=board,
        imageSize=img_gray.shape,
        cameraMatrix=None,
        distCoeffs=None
    )

    # Calculamos el error de reproyeccion 
    mean_error = 0
    for i in range(len(corners_list)):
        img_points, _ = cv2.projectPoints(
            objectPoints=board.chessboardCorners,
            rvec=rvecs[i],
            tvec=tvecs[i],
            cameraMatrix=cameraMatrix,
            distCoeffs=distCoeffs
        )
        # Calculamos el RMSE para cada punto estudiado
        error = cv2.norm(corners_list[i], img_points, cv2.NORM_L2) / len(img_points)
        mean_error += error
    
    # Hacemos la media de los errores obtenidos
    print(f'\nReprojection error: {mean_error / len(corners_list)}')

    return ret, cameraMatrix, distCoeffs, rvecs, tvecs

# Funcion para buscar los vertices de una figura
def detectar_vertices(imagen):
    # Pasamos la imagen a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Aplicamos un filtro Canny para detectar los bordes
    canny = cv2.Canny(gray, 10, 150)
    # Dilatamos y erosionamos (closing) para eliminar los falsos positivos
    canny = cv2.dilate(canny, None, iterations=1)
    canny = cv2.erode(canny, None, iterations=1)
    # Buscamos contornos dentro de la imagen
    contornos, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Contamos cuantos vertices tiene cada contorno
    vertices_poligonos = [cv2.approxPolyDP(contorno, 0.02 * cv2.arcLength(contorno, True), True) for contorno in contornos]
    return vertices_poligonos


# Funcion para detectar si una figura es una estrella
def detectar_estrella(imagen, vertices):
    # Partiendo de los vertices encontrados, calculamos los lados y angulos de la figura
    lados = [np.array([vertices[(i + 1) % 10][0][0], imagen.shape[0] - vertices[(i + 1) % 10][0][1]]) - np.array([vertices[i][0][0], imagen.shape[0] - vertices[i][0][1]]) for i in range(len(vertices))]
    angulos = [np.degrees(np.arccos(np.dot(lados[i], lados[(i + 1) % len(lados)]) / (np.linalg.norm(lados[i]) * np.linalg.norm(lados[(i + 1) % len(lados)]))))for i in range(len(lados))]
    # Verificamos que se cumple la secuencia de angulo agudo - angulo obtuso
    check = [False if (angulos[i] < 90 and angulos[(i + 1) % len(angulos)] < 90) or (angulos[i] > 90 and angulos[(i + 1) % len(angulos)] > 90) else True for i in range(len(angulos))]
    if all(check): # Si se cumple, es una estrella
        return True
    return False


# Funcion para detectar si una figura es un cuadrado
def detectar_cuadrado(imagen, vertices): 
    # Partiendo de los vertices encontrados, calculamos los lados y angulos de la figura
    lados = [np.array([vertices[(i + 1) % 4][0][0], imagen.shape[0] - vertices[(i + 1) % 4][0][1]]) - np.array([vertices[i][0][0], imagen.shape[0] - vertices[i][0][1]]) for i in range(len(vertices))]
    angulos = [np.degrees(np.arccos(np.dot(lados[i], lados[(i + 1) % len(lados)]) / (np.linalg.norm(lados[i]) * np.linalg.norm(lados[(i + 1) % len(lados)]))))for i in range(len(lados))]
    # Verificamos que todos los angulos son rectos
    check = [85 <= angulos[i] <= 95 for i in range(len(angulos))]
    if all(check):
        # Si todos los lados son iguales, es un cuadrado
        lados_iguales = [np.isclose(np.linalg.norm(lados[i]), np.linalg.norm(lados[(i + 1) % len(lados)]), atol = 20) for i in range(len(lados))]
        if all(lados_iguales):
            return True
    return False


# Funcion para detectar si una figura es un pentagono
def detectar_pentagono(imagen, vertices):
    # Teniendo los vértices, podemos sacar los lados de la figura
    lados = [np.array([vertices[(i + 1) % 5][0][0], imagen.shape[0] - vertices[(i + 1) % 5][0][1]]) - np.array([vertices[i][0][0], imagen.shape[0] - vertices[i][0][1]]) for i in range(len(vertices))]
    # Comprobamos que los lados son iguales
    lados_iguales = [np.isclose(np.linalg.norm(lados[i]), np.mean(np.array([np.linalg.norm(lado) for lado in lados])), atol = 15) for i in range(len(lados))]
    # Si todos los lados son iguales, es un pentágono
    if all(lados_iguales):
        return True
    return False


# Funcion para detectar la secuencia de figuras
def secuencia(imagen, paso):
    # Buscamos los vertices de la figura
    vertices_poligonos = detectar_vertices(imagen)
    if len(vertices_poligonos) == 1:
        vertices = vertices_poligonos[0]
        # Comprobamos que la figura con 10 vertices es una estrella
        if len(vertices) == 10 and paso == 0:
            if detectar_estrella(imagen, vertices):
                return True, '\nEstrella detectada'
        # Comprobamos que la figura con 4 vertices es un cuadrado
        elif len(vertices) == 4 and paso == 1:
            if detectar_cuadrado(imagen, vertices):
                return True, '\nCuadrado detectado'
        # Comprobamos que la figura con 5 vertices es un pentagono
        elif len(vertices) == 5 and paso == 2:
            if detectar_pentagono(imagen, vertices):
                return True, '\nPentágono detectado'
        else:
            # Si se introduce una figura en orden incorrecto, se reinicia el proceso
            return False, 'Contraseña incorrecta. Vuelva a intentarlo'
    # Si se introduce más de una figura, se reinicia el proceso
    return False, 'Por favor, introduzca la contraseña de figura en figura.'


# Funcion para detectar si hay alguna postura de yoga de nuestra base de datos en la imagen
def identificacion_plantilla(imagen, plantillas, keypoints_plantillas):
    # Buscamos los keypoints y descriptores de la imagen a analizar
    keypoints_busqueda, descriptores_busqueda = sift.detectAndCompute(imagen, None)
    if keypoints_busqueda is not None and descriptores_busqueda is not None and len(descriptores_busqueda) > 1:
        # Buscamos las coincidencias entre los keypoints de la imagen a analizar y los keypoints de todas las plantillas
        matches = [bf.knnMatch(descriptores_plantilla, descriptores_busqueda, k=2) for keypoints_plantilla, descriptores_plantilla in keypoints_plantillas]
        buenas_coincidencias = [[m for m, n in matches1 if len(matches1) > 1 and m.distance < 0.75 * n.distance and m is not None and n is not None] for matches1 in matches ]
        maximos_puntos = max([len(elemento) for elemento in buenas_coincidencias])
        # Si el numero de coincidencias es mayor o igual a 25, se considera que se ha encontrado una postura de yoga
        if maximos_puntos >= 25:  
            # Buscamos el indice de la plantilla con mayor numero de coincidencias
            indice = [i for i, j in enumerate(buenas_coincidencias) if len(j) == maximos_puntos][0]
            return indice 
    return None
   
   
if __name__ == "__main__":
    modo = input('\n1 si quiere calibrar una nueva cámara. \n2 si quiere utilizar la última cámara calibrada. \nPulse una de las siguientes opciones: ') 
    if int(modo) == 1:
        # Parametros para la calibracion
        IMAGES_DIR = 'imagenes_calibracion/'
        IMAGES_FORMAT = 'jpg'
        # Dimensiones en cm
        MARKER_LENGTH = 2.07
        SQUARE_LENGTH = 3.5

        # Calibramos haciendo uso de la funcion calibrar_charuco
        ret, cameraMatrix, distCoeffs, rvecs, tvecss = calibrar_charuco(IMAGES_DIR, IMAGES_FORMAT, MARKER_LENGTH, SQUARE_LENGTH)

        # Guardamos los coeficientes en un fichero de texto para poder utilizarlos en el futuro
        np.savetxt('distCoeffs.txt', distCoeffs)
        np.savetxt('matrix.txt', cameraMatrix)
        
        print('\nCámara calibrada correctamente')
        
    elif int(modo) == 2: 
        # Leemos los parámetros de la cámara calibrada
        distCoeffs = np.loadtxt('distCoeffs.txt')
        cameraMatrix = np.loadtxt('matrix.txt')
    
    # Inicializamos la cámara
    picam = Picamera2()
    picam.preview_configuration.main.size=(500,300)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    
    print('\n Para iniciar el programa introduzca la contraseña correcta. Pulsa ENTER para registrar cada patrón.')
    
    # Comenzamos el sistema de seguridad
    # Se piden 3 figuras en orden: estrella, cuadrado y pentagono
    paso = 0
    while paso < 3:
        frame = picam.capture_array()
        cv2.imshow('imagen', frame)
        # Mostramos el ideo y cuando se le da a la tecla ENTER se hace captura la imagen para analizarla
        if cv2.waitKey(1) & 0xFF == 13:
            # Buscamos el patron que tiene que introducir el usuario
            patron_reconocido, mensaje = secuencia(frame, paso)
            # Si se ha reconocido el patron, se pasa al siguiente paso
            if patron_reconocido:
                print(mensaje)
                print('Por favor, introduzca la siguiente figura.')
                paso += 1
            else:
                print(mensaje)
                paso = 0
        # Para salir del programa
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Comenzamos el sistema de identificacion de posturas de yoga
    # Si se ha introducido la contraseña correctamente, se activa la guia de posturas de yoga
    if paso == 3:
        print('\n¡CONTRASEÑA CORRECTA! Activando la guía de posturas de yoga ...')
        
        # Cargamos las plantillas       
        filenames = glob.glob('plantillas/*.png')
        plantillas = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in filenames]
        
        # Cargamos la información de cada postura
        info_txt = glob.glob('Info/*.txt')
        info  = [open(ruta, 'r').read() for ruta in info_txt]
        
        # Buscamos los keypoints y descriptores de cada plantilla
        keypoints_plantillas = [sift.detectAndCompute(plantilla, None) for plantilla in plantillas]

        
        indice_antes = None   
        while True:
            frame = picam.capture_array()
            cv2.imshow('imagen', frame)
            # Quitamos la distorsion de la imagen capturada con los parametros de la camara calibrada
            imagen_dst = cv2.undistort(frame, cameraMatrix, distCoeffs, None, cameraMatrix)
            # Buscamos si hay alguna postura de yoga en la imagen
            indice = identificacion_plantilla(frame, plantillas, keypoints_plantillas)
            if indice != None:
                # Si se ha encontrado una postura y es distinta a la anterior, se muestra la informacion de la postura
                if indice_antes != indice:
                    nombre = filenames[indice][11:-4]
                    print(nombre)
                    print('\nInformación acerca de la siguiente postura: ' + nombre + '\n')
                    print(info[info_txt.index(f'Info/{nombre}.txt')])
                indice_antes = indice
            # Para salir del programa
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
        # Cerramos la ventana y finalizamos el programa   
        cv2.destroyAllWindows()
        print('Se ha finalizado la ejecución del programa')
    # Si no se ha introducido la contraseña correctamente y el usuario ha decidido cerrar, se finaliza el programa
    else: 
        cv2.destroyAllWindows()
        print('Se ha finalizado la ejecución del programa')  