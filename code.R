
# OBJETIVO ----------------------------------------------------------------

## El objetivo es construir un trabajo completo sobre la clasificación de sentimientos de películas sobre el famoso IMDB dataset. Las partes del proyecto van a ser:
## 1. Comprender y preparar los sets
## 2. Elección del approach
## 3. Preparación de los datos para ser introducidos a la red
## 4. Entrenamiento del modelo
## 5. Preparación de predicciones sobre muestras fuera del set

##adicionalmente, pese a elegir el approach que nos diga el enfoque (BOW), también entrenaremos un modelo secuencial.

# 1. COMPRENDER Y PREPARAR LOS SETS --------------------------------------

library(tidyverse)
library(tensorflow)
library(keras)
library(tfdatasets)
library(fs)
## descarga y visualizacion de la estructura de los datos y de su contenido
setwd("C:/Users/adria/Desktop/PROYECTOS/imdb classification - R")
url <- "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filename <- basename(url)
download.file(url, destfile = filename)
untar(filename) #descomprimir 
fs::dir_tree("aclImdb", recurse = 1, type = "directory") #observamos la estructura del directorio
fs::dir_delete("aclImdb/train/unsup/")
length(list.files("C:/Users/adria/Desktop/PROYECTOS/imdb classification - R/aclImdb/test/pos"))
length(list.files("C:/Users/adria/Desktop/PROYECTOS/imdb classification - R/aclImdb/test/neg"))
length(list.files("C:/Users/adria/Desktop/PROYECTOS/imdb classification - R/aclImdb/train/pos"))
length(list.files("C:/Users/adria/Desktop/PROYECTOS/imdb classification - R/aclImdb/train/neg"))
## hay 25k reviews para entrenamiento y 25k para testeo
writeLines(readLines("C:/Users/adria/Desktop/PROYECTOS/imdb classification - R/aclImdb/test/pos/1_10.txt", warn = FALSE)) #observamos una de las reviews
##creamos un dataset de validacion con el 20% aprox de los archivos del directorio de entrenamiento
set.seed(1337) # fijamos una semilla para que los archivos de validacion extraidos siempre sean los mismos
base_dir <- path("aclImdb")
for (category in c("neg", "pos")) {
filepaths <- dir_ls(base_dir / "train" / category)
num_val_samples <- round(0.2 * length(filepaths))
val_files <- sample(filepaths, num_val_samples)
dir_create(base_dir / "val" / category)
file_move(val_files,
base_dir / "val" / category)
}
length(list.files("C:/Users/adria/Desktop/PROYECTOS/imdb classification - R/aclImdb/val/pos"))
length(list.files("C:/Users/adria/Desktop/PROYECTOS/imdb classification - R/aclImdb/val/neg"))
##cargar en el environment los archivos de cada set en un formato adecuado, esto es muy sencillo gracias a keras: text_dataset_from_directory()
train_ds <- text_dataset_from_directory("aclImdb/train")
val_ds <- text_dataset_from_directory("aclImdb/val")
test_ds <- text_dataset_from_directory("aclImdb/test")
c(inputs, targets) %<-% iter_next(as_iterator(train_ds))
#observamos ya la primera observacion cargada
inputs[1]
targets[1]

# 2. ELECCIÓN DEL APPROACH ------------------------------------------------
## La elección se basa en la intuición experimental de investigadores de deep learning en google y puede ser vista aqui (http://mng.bz/AOzK)
## Basicamente, consiste en un ratio entre numero de muestras / media de longitud de las muestras con el limite de 1500. Si el ratio supera
## los 1500, es mejor la elección de un modelo secuencial, puesto que el tamaño muestral es muy grande y las secuencias son cortas y si es menor,
## se elige un modelo bag of words o el clásico approach basado unicamente en frecuencias de tokens y no en orden de ML.

training_samples <- length(list.files("C:/Users/adria/Desktop/PROYECTOS/imdb classification - R/aclImdb/train/pos")) +
  length(list.files("C:/Users/adria/Desktop/PROYECTOS/imdb classification - R/aclImdb/train/neg"))
#definimos un bucle que nos calcule la media de caracteres que tienen las criticas
caracteres_medios <- list()
caracteres <- list()
replicate(100,{
  c(inputs, targets) %<-% iter_next(as_iterator(train_ds))
   for (i in 1:length(inputs)) {
     caracteres[[i]] <- nchar(as.array(inputs[i]))
     caracteres_medios[[i]] <- mean(unlist(caracteres))
   }
  caracteres_medios[[i]] <- mean(unlist(caracteres))
})
longitud_media <- mean(unlist(caracteres_medios))  
training_samples / longitud_media
#teniendo en cuenta el ratio muy bajo, vamos a utilizar el BOW approach en nuestra red


# 3. PREPARACIÓN DE LOS DATOS PARA SER INTRODUCIDOS A LA RED ---------------------------------------
## puesto que es un conjunto de datos sobradamente trabajado, vamos a ir directamente con bigrams como tokens y en lugar de un one hot encoding, un tf-idf encoding para cada
## dimension o token

text_only_train_ds <- train_ds %>% #guardamos solo las criticas, no la label
dataset_map(function(x, y) x)

text_vectorization <- #definimos la vectorizacion del texto, con bigramas, las 20k palabras mas frecuentes (enfoque aceptado en la comunidad) y el tf_idf
layer_text_vectorization(ngrams = 2,
max_tokens = 20000,
output_mode = "tf_idf")

adapt(text_vectorization, text_only_train_ds) #adaptamos el texto a estas especificaciones


binary_2gram_train_ds <- train_ds %>%
dataset_map( ~ list(text_vectorization(.x), .y),
num_parallel_calls = 4)
binary_2gram_val_ds <- val_ds %>%
dataset_map( ~ list(text_vectorization(.x), .y),
num_parallel_calls = 4)
binary_2gram_test_ds <- test_ds %>%
dataset_map( ~ list(text_vectorization(.x), .y),
num_parallel_calls = 4)

#podemos observar la estructura del texto formateado para que pueda entrar en la red
c(inputs, targets) %<-% iter_next(as_iterator(binary_2gram_train_ds))
str(inputs)
str(targets)

# 4. ENTRENAMIENTO DEL MODELO (RED) ---------------------------------------
callbacks <- list(callback_model_checkpoint("tfidf_2gram.keras",
save_best_only = TRUE)) #vamos a guardar el modelo que mejor performance tenga en la epoch correspondiente para su posterior uso

model <- keras_model_sequential() %>%
layer_dense(16, activation = "relu", input_shape = c(20000)) %>%
layer_dropout(0.5) %>%
layer_dense(1, activation = "sigmoid")

model %>% compile(optimizer = "rmsprop",
loss = "binary_crossentropy",
metrics = "accuracy")

model %>% fit(
dataset_cache(binary_2gram_train_ds),
validation_data = dataset_cache(binary_2gram_val_ds),
epochs = 10,
callbacks = callbacks
)


model <- load_model_tf("tfidf_2gram.keras")
evaluate(model, binary_2gram_test_ds)["accuracy"] %>%
sprintf("Test acc: %.3f", .) %>% cat("\n") #86.7%


# 5. PREDICCIONES FUERA DEL SET -------------------------------------------
## con el modelo entrenado y cargado en consola, podemos prepararnos para cualquier review
## la predicción es sencilla, se trata de crear un modelo (functional API) que aplique a un tensor cadena la vectorizacion de nuestro modelo (ultima cargada)
## y nuestro modelo (ultimo cargado)

## creacion del modelo
inputs <- layer_input(shape = c(1), dtype = "string")
outputs <- inputs %>%
text_vectorization() %>%
model()
inference_model <- keras_model(inputs, outputs)
## creacion de la critica
raw_text_data <- "Infinity war is one of the best MCU protects. It has a great story, great acting, and awesome looking. If you aren't a Marvel fan or haven't watched most of the previous MCU movies this however, won't be something for you. Let's start with Thanos, definitely one of the best villains, he has a motive, is well played, you can even say that Infinity war tells his story and not the story of a hero. But also most of the other cast members were great in their role and again, if you love Marvel, watch this movie." %>%
as_tensor(shape = c(-1, 1))
## predicción
predictions <- inference_model(raw_text_data)
cat(sprintf("%.2f percent positive\n",
as.numeric(predictions) * 100))

#OBJETIVO 2: UTILIZAR UN MODELO SECUENCIAL Y COMPARAR RESULTADOS EN TERMINOS DE ACCURACY Y DE RENDIMIENTO COMPUTACIONAL.
max_length <- 600 #elegimos un maximo de tokens (en este caso de palabras) que compongan las secuencias ya que el approach
#es mucho mas intenso computacionalmente
max_tokens <- 20000 # al igual que antes, las 20000 palabras mas frecuentes

#definimos ahora la vectorizacion, en este caso con los tokens por defecto (palabras) asignando un numero entero a las 20k palabras
#de la secuencia
text_vectorization <- layer_text_vectorization(
max_tokens = max_tokens,
output_mode = "int",
output_sequence_length = max_length
)
text_only_train_ds <- train_ds %>% #guardamos solo las criticas, no la label (exactamente igual que antes)
dataset_map(function(x, y) x)

adapt(text_vectorization, text_only_train_ds)

int_train_ds <- train_ds %>% dataset_map( ~ list(text_vectorization(.x), .y),num_parallel_calls = 4)
int_val_ds <- val_ds %>% dataset_map( ~ list(text_vectorization(.x), .y),num_parallel_calls = 4)
int_test_ds <- test_ds %>% dataset_map( ~ list(text_vectorization(.x), .y),num_parallel_calls = 4)

#podemos observar, de nuevo, la estructura del texto formateado para que pueda entrar en la red
c(inputs, targets) %<-% iter_next(as_iterator(int_train_ds))
str(inputs)
str(targets)

#podemos ahora entrenar un modelo secuencial (bidireccional lstm sencilla)

callbacks <- list(callback_model_checkpoint("lstm_gram.keras",
save_best_only = TRUE)) #vamos a guardar el modelo que mejor performance tenga en la epoch correspondiente para su posterior uso

model <- keras_model_sequential() %>%
layer_embedding(input_dim = max_tokens,output_dim = 256,mask_zero = TRUE) %>% 
bidirectional(layer_lstm(units = 32)) %>%
layer_dropout(0.5) %>%
layer_dense(1, activation = "sigmoid") #es un modelo con muchos mas parametros


model %>% compile(optimizer = "rmsprop",
loss = "binary_crossentropy",
metrics = "accuracy")

model %>% fit(
dataset_cache(int_train_ds),
validation_data = dataset_cache(int_val_ds),
epochs = 10,
callbacks = callbacks
)

model <- load_model_tf("lstm_gram.keras")
evaluate(model, int_test_ds)["accuracy"] %>%
sprintf("Test acc: %.3f", .) %>% cat("\n") #84.7



