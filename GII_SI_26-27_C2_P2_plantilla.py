# DIFICULTADES ENCONTRADAS:
# 1. 
# Instalar tensorflow en mi portatil con wsl. Dios que barbaridad, he tenido que actializar todos los paquetes de Python
# manualmente y conectarlo a WSL: Ubuntu, reinstalando todas las extensiones y metiendo mil comandos en la terminak de wsl.
# Para hacer esto he tenido que usar la IA porque no me aclaro con la instalacion de paquetes y tal porque hay mil maneras
# y sino es imposible.


# PARA TRABAJAR AQUÍ SIEMPRE TENGO QUE TENER EL BOTÓN DE ABAJO A LA IZQUIERDA CONECTADO A WSL: Ubuntu, sino no podré ejecutar nada.


# EJECUTAR ESTO SI QUIERO EJECUTAR SCRIPTS MANUALMENTE DESDE UNA TERMINAL WSL NUEVA
# source ~/tensorflow/bin/activate
# (No es necesario que esté ubicado en ningún directorio para ejecutar eso)
# ah bueno y el tensorflow debe de ser el interprete de mi entorno virtual: comprobar que siempre está en la esquina inferior derecha


import logging
logging.disable(logging.WARNING)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import cifar10
from keras.layers import Input, Flatten, Rescaling, Dense, Activation, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# para importar las imagenes de Keras y ver en qué consisten
import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(x_train.shape, x_train.dtype)
print(y_train.shape, y_train.dtype)
print(x_test.shape, x_test.dtype)
print(y_test.shape, y_test.dtype)

# EXPLICACIÓN DE x_train y y_train
# x_train = es un array de 50.000 imagenes de 32x32 pixeles. Cada pixel es una caracteristica de una observacion (no se muy bien que significa eso ultimo)
# y_train = es otro array de 50.000 numeros que estan asociados a cada categoria (0: avion, 1: coche, 2: pajaro, etc) y querremos que MLP nos devuelva estos valores cuando este bien entrenado.
# x_text y y_text es lo mismo pero contienen 10.000 observaciones cada uno.
# (CREO QUE LAS OBSERVACIONES SON CADA IMAGEN)


# show_image: Esta es la funcion que sale en el doc, que explica que con imshow de plt se pueden mostrar imagenes
def show_image(imagen, titulo):
    plt.figure()
    plt.suptitle(titulo)
    plt.imshow(imagen, cmap= "Greys")
    plt.show()

from random import sample # supongo que hay que cambiar esta parte en vez de random será dog

for i in sample(list(range(len(x_train))), 3):
    titulo = "Mostrando imagen x_train[" + str(i) + "]"
    titulo = titulo + "-- y_train[" + str(i) + "] = " + str (y_train[i])
    show_image(x_train[i], titulo)

from random import sample # supongo que hay que cambiar esta parte en vez de random será dog

for i in sample(list(range(len(x_train))), 3):
    titulo = "Mostrando imagen x_train[" + str(i) + "]"
    titulo = titulo + "-- y_train[" + str(i) + "] = " + str (y_train[i])
    show_image(x_train[i], titulo)



# PARA MOSTRAR AL PERRO, con imshow
def mostrar_perro(x_train, y_train):
    indices_perros = np.where(y_train == 5)[0]  # np.where entiendo que es de asignación

    # indice1 es el primer indice encontrado
    indice1 = indices_perros[0]

    # extraemos la imagen
    imagen_perro = x_train[indice1]

    # esto genera la visualizacion
    plt.figure()
    plt.suptitle("Mostrando perro")
    plt.imshow(imagen_perro)
    plt.show()



#
# FUNCIONES PARA HACER LAS GRÁFICAS
#

def plot_history(history, title=None, show=True, save_file_name=None):
    '''
        * history: el objeto history que devuelve model.fit()
        * title: texto para el título de la figura
        * show: ponlo a False para que no se muestre la figura por pantalla
            Si está a True, utiliza plt.show(block=False), al final de la función, para que
                no se pare la ejecución mientras se muestra la imagen
            Si deseas que se creen las imágenes y queden abiertas mientras sigue la ejecución,
                tendrás que devolver las figuras a la función principal para que no "mueran" al
                salir del scope actual.
        * save_file_name: nombre del archivo donde guardar la figura (ha de ser un .pdf)
    '''

    accuracy = history[0]
    loss = history[1]
    val_accuracy = history[2]
    val_loss = history[3]

    # fig cubre toda la figura, ax1 solo la parte entre ejes
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')

    ax1.set_ylabel('Accuracy', color='r')
    ax1.plot(accuracy, 'r--', label='train_acc')   # r--: red line
    ax1.plot(val_accuracy, 'r-', label='val_acc')  # r-: red dashed line
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='b')
    ax2.plot(loss, 'b--', label='train_loss')
    ax2.plot(val_loss, 'b-', label='val_loss')
    ax2.tick_params(axis='y', labelcolor='b')

    plt.title((title if title else "")+' Training history ')
    fig.legend(loc='center left', bbox_to_anchor=(0.66, 0.52))  # mostrar leyenda
    fig.tight_layout()  # para que no queden recortadas algunas de las etiquetas
    
    if save_file_name:
        plt.savefig(save_file_name)
    if show:
        plt.show(block=True)
    plt.close()


def plot_grouped_bars(results, errors, xlabels, title, y1log=False, y2log=False, show=True, save_file_name=None):
    '''
        * results: lista con los valores (medios) de accuracy y tiempo
        * errors: lista con los márgenes de error (max-media y media-min) de accuracy y tiempo
        * xlabels: lista con las etiquetas del eje x
        * title: texto para el título de la figura
        * y1log e y2log: ponlos a True para emplear escala logarítmica
        * show: ponlo a False para que no se muestre la figura por pantalla
        * save_file_name: nombre del archivo donde guardar la figura (ha de ser un .pdf)
    '''

    # Preparar los valores a mostrar en un dataframe
    df = pd.DataFrame(results, columns=['accuracy', 'time'])
    ax = df.plot(kind='bar', secondary_y=['time'], rot=0, yerr=errors)
    ax.right_ax.set_ylabel('time')
    ax.set_ylabel('accuracy')

    # Calcular mínimo del eje izquierdo (accuracy)
    min_acc = df['accuracy'].min()
    min_y1 = np.floor(min_acc*10)/10
    if min_y1 + 0.05 > min_acc:
        min_y1 = max(0, min_y1 - 0.05)
    ax.set_ylim(bottom=min_y1)

    # Calcular mínimo del eje derecho (time, que a menudo será 0)
    min_t = df['time'].min()
    min_y2 = np.floor(min_t/5)*5
    if min_y2 + 10 > min_t:
        min_y2 = max(0, min_y2 - 10)
    ax.right_ax.set_ylim(bottom=min_y2)

    if y1log:
        ax.set_yscale('log')
    if y2log:
        ax.right_ax.set_yscale('log')
    ax.set_xticks(range(len(results)), xlabels)
    plt.title(title)
    # Si no coloca la leyenda en un buen lugar puedes fijarla tú aquí: 
    #ax.legend(loc='lower right')
    
    if save_file_name:
        plt.savefig(save_file_name)
    if show:
        plt.show(block=True) #utiliza block=False para que no se pare la ejecución
        # Si deseas que se creen las imágenes y queden abiertas mientras sigue la ejecución,
        # tendrás que hacer return del plot a la función principal para que no "mueran" al salir
    plt.close()


#
# FUNCIONES GENERALES PARA ENTRENAR MLPs Y MOSTRAR LOS RESULTADOS
#

def load_cifar10():

    # Cargar los datos, divididos entre entrenamiento y test
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Escalar las imagenes al rango [0,1] nada más cargar es lo típico:
    #x_train = np.asarray(x_train) / 255.0
    #x_test = np.asarray(x_test) / 255.0
    # Pero nosotros vamos a escalar diferente según la activación, usando una capa en el modelo

    # Convertir las etiquetas a matrices categóricas
    y_train = keras.utils.to_categorical(np.asarray(y_train), 10)
    y_test = keras.utils.to_categorical(np.asarray(y_test), 10)

    return (x_train, y_train), (x_test, y_test)


def defntest_model(x_train, y_train, x_test, y_test, batch_size=128, verbose=True,
                   layers=[("Flaten",), ("Dense", 128), ("sigmoid", ), ("Dense", 64, ), ("sigmoid", ), ],
                   scaling=[0, 1], kernel_initializer="glorot_uniform",
                   epochs=50, learning_rate=0.001, patience=0, repetitions=1,
                   model_file_name=None, fig_file_name=None, show_fig=True, label=None):
    '''
    El parámetro layers es una lista con una tupla para cada capa
    Realiza [repetitions] entrenamientos independientes (modelos creados de cero)
    Acumula y promedia el history para su ploteado, devuelve las tasas de acierto y tiempos min, max y avg
    Nota: el promediado de los valores de accuracy y tiempo es útil para comparar entre modelos, y el
          promediado de los histories es útil para estimar el número de épocas ideales pero es engañoso
          para evaluar la calidad de un entrenamiento ya que esconde los posibles vaivenes de accuracy y loss
          Idealmente, se dibujaría el history promedio y uno o más de los individuales para informar mejor.
    '''

    assert(repetitions >= 1)
    
    activaciones = ["sigmoid", "tanh", "relu", "leaky_relu", "gelu", "selu", "mish", "swish", ]
    histories = []
    total_time = 0.0
    total_score = 0.0
    min_time = 9999.9
    min_score = 9999.9
    max_time = 0.0
    max_score = 0.0
    
    first = True
    for _ in range(repetitions):
        # Iniciar un modelo secuencial, siempre con una capa de entrada y otra de reescalado
        model = keras.models.Sequential()
        model.add(Input(x_train[0].shape))
        model.add(Rescaling(scale=(scaling[1]-scaling[0])/255.0, offset=scaling[0]))
        # Añadir el resto de capas descritas por la configuración que se le pasa
        for layer in layers:
            if layer[0] == "Flaten":
                model.add(Flatten())
            elif layer[0] == "Dense":  
                model.add(Dense(layer[1], kernel_initializer=kernel_initializer))
            elif layer[0] == "BN":
                model.add(BatchNormalization())
            elif layer[0] == "DO":
                model.add(Dropout(layer[1]))
            elif layer[0] in activaciones:
                model.add(Activation(layer[0]))
            elif layer[0] == "Conv2D":
                model.add(Conv2D(layer[1], kernel_size=(3, 3)))
            elif layer[0] == "MaxPooling2D":
                model.add(MaxPooling2D(pool_size=layer[1]))
            else:
                print("Parámetro desconocido en configuraciones: "+str(layer[0]))
        # Cerrar siempre con la capa de salida softmax
        model.add(Dense(10, activation="softmax"))

        model.compile(loss="categorical_crossentropy", metrics=["accuracy"],
                      optimizer=Adam(learning_rate=learning_rate))

        if first:
            model.summary()
            first = False

        callbacks = []
        if patience>0:
            callbacks.append([EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)])

        start = time.time()
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                            validation_split=0.1, callbacks=callbacks, verbose=verbose)
        fit_time = time.time() - start

        test_score = model.evaluate(x_test, y_test, verbose=verbose)
        print("Test accuracy, test loss and fit time:", test_score[1], test_score[0], fit_time)

        # Acumular histories para promediar antes de plotear
        histories.append(np.asarray(list(history.history.values())))
        
        if min_time > fit_time:
            min_time = fit_time
        if max_time < fit_time:
            max_time = fit_time

        if min_score > test_score[1]:
            min_score = test_score[1]
        if max_score < test_score[1]:
            max_score = test_score[1]
        
        total_time = total_time + fit_time
        total_score = total_score + test_score[1]

    # Obtener la longitud máxima de las historias
    max_epocas = max(arr.shape[1] for arr in histories)
    # Rellenar las historias con el último valor para que todas tengan la misma longitud 
    arrays_padded = [np.pad(arr, ((0, 0), (0, max_epocas - arr.shape[1])), mode='edge') for arr in histories]
    # Promediar historias
    historia_promedio = np.mean(np.stack(arrays_padded), axis=0)
    
    plot_history(historia_promedio, label, show=show_fig, save_file_name=fig_file_name)

    if model_file_name:
        model.save(model_file_name)
    
    return (min_score, total_score/repetitions, max_score), (min_time, total_time/repetitions, max_time)
    

def train_many_and_plot(x_train, y_train, x_test, y_test, configs, repetitions=1, title="Results", save_file_name=None):
    '''
    Entrena múltiples modelos, en función de la configuración que se le pasa, y muestra los resultados en una gráfica de barras.
    
    Formato para una configuración (ver ejemplos en el ejercicio 2):
        {
            "layers": [("nombre_capa", parámetros opcionales según el tipo de capa), ...] 
            "scaling": [valor_inferior, valor_superior],
            "kernel_initializer": "nombre_inicialzador", 
            "batch_size": entero,
            "learning_rate": real, 
            "epochs": entero,
            "label": "título", # para la gráfica de evolución del entrenamiento y la etiqueta del eje X en la gráfica de barras
            "fig_file_name": "fichero.pdf", # nombre del archivo donde guardar la gráfica de evolución del entrenamiento 
        }
    '''
    results = []
    avgs_acc = []
    mins_acc = []
    maxs_acc = []
    avgs_t = []
    mins_t = []
    maxs_t = []
    xlabels = []

    for cfg in configs: 
        accs, times = defntest_model(x_train, y_train, x_test, y_test, **cfg, repetitions=repetitions)
        mins_acc.append(accs[0])
        avgs_acc.append(accs[1])
        maxs_acc.append(accs[2])
        mins_t.append(times[0])
        avgs_t.append(times[1])
        maxs_t.append(times[2])
        results.append((accs[1], times[1]))
        xlabels.append(cfg["label"])

    errors = {
        'accuracy': [ np.array(avgs_acc) - np.array(mins_acc), np.array(maxs_acc) - np.array(avgs_acc) ],
        'time': [ np.array(avgs_t) - np.array(mins_t), np.array(maxs_t) - np.array(avgs_t) ]
    }

    plot_grouped_bars(results, errors, xlabels, title, show=True, save_file_name=save_file_name)


#
# FUNCIONES PARA LOS EJERCICIOS
#

def ejercicio1(x_train, y_train, x_test, y_test):

    defntest_model(x_train, y_train, x_test, y_test, epochs=100,
                   layers=[("Flaten",), ("Dense", 128), ("sigmoid", ), ("Dense", 64, ), ("sigmoid", ), ],
                   fig_file_name="128+64_Sigm_EP100.pdf")


def ejercicio2(x_train, y_train, x_test, y_test):

    configs = [
        {
            "layers": [("Flaten", ), ("Dense", 128), ("sigmoid", ), ("Dense", 128), ("sigmoid", )],
            "scaling": [0, 1],
            "kernel_initializer": "glorot_uniform",
            "batch_size": 16,
            "learning_rate": 0.001,
            "label": "BS 16",
            "fig_file_name": "2x128_Sigm_BS16.pdf",
        },
        {
            "layers": [("Flaten", ), ("Dense", 128), ("sigmoid", ), ("Dense", 128), ("sigmoid", )],
            "scaling": [0, 1],
            "kernel_initializer": "glorot_uniform",
            "batch_size": 128,
            "learning_rate": 0.001,
            "label": "BS 128",
            "fig_file_name": "2x128_Sigm_BS128.pdf",
        },
        {
            "layers": [("Flaten", ), ("Dense", 128), ("sigmoid", ), ("Dense", 128), ("sigmoid", )],
            "scaling": [0, 1],
            "kernel_initializer": "glorot_uniform",
            "batch_size": 1024,
            "learning_rate": 0.001,
            "label": "BS 1024",
            "fig_file_name": "2x128_Sigm_BS1024.pdf",
        },
    ]
    train_many_and_plot(x_train, y_train, x_test, y_test, configs, title="2x128 Sigmoid, variando BS",
                              save_file_name="comp_2x128_Sigm_BS.pdf")

    configs = [
        {
            "layers": [("Flaten", ), ("Dense", 128), ("sigmoid", ), ("Dense", 128), ("sigmoid", )],
            "scaling": [0, 1],
            "kernel_initializer": "glorot_uniform",
            "batch_size": 256,
            "learning_rate": 0.05,
            "label": "LR 0.05",
            "fig_file_name": "2x128_Sigm_LR05.pdf",
        },
        {
            "layers": [("Flaten", ), ("Dense", 128), ("sigmoid", ), ("Dense", 128), ("sigmoid", )],
            "scaling": [0, 1],
            "kernel_initializer": "glorot_uniform",
            "batch_size": 256,
            "learning_rate": 0.001,
            "label": "LR 0.001",
            "fig_file_name": "2x128_Sigm_LR001.pdf",
        },
        {
            "layers": [("Flaten", ), ("Dense", 128), ("sigmoid", ), ("Dense", 128), ("sigmoid", )],
            "scaling": [0, 1],
            "kernel_initializer": "glorot_uniform",
            "batch_size": 256,
            "learning_rate": 0.00002,
            "label": "LR 0.00002",
            "fig_file_name": "2x128_Sigm_LR00002.pdf",
        },
    ]
    train_many_and_plot(x_train, y_train, x_test, y_test, configs, title="2x128 Sigmoid, variando LR",
                              save_file_name="comp_2x128_Sigm_LR.pdf")




# ESTA ES LA FUNCIÓN PARA EJECUTAR, aqui se añaden las llamadas a las funciones

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_cifar10()

    ejercicio1(x_train, y_train, x_test, y_test)
    # El primer ejercicio experimenta con un solo modelo de características fijas
    # por lo que la parte de experimentación está completa con el código de ejercicio1()
    # Sin embargo, ejercicio2() es solo una sugerencia de lo que se puede probar
    # y los modelos que prueba son ilustrativos de los que se pueden explorar
    #ejercicio2(x_train, y_train, x_test, y_test)


    # Para mostrar al perro
    mostrar_perro(x_train, y_train)
