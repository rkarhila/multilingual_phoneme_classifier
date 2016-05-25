# multilingual_phoneme_classifier
Preprocessing, training and test scripts

# How-to

1. Pay a silly amount of money to get speecon corpus
2. Preprocess with the scripts in `data_preprocessing` 
3. Then:

  ```
  cd model_training
  virtualenv keras-tensorflow
  source keras-tensorflow/bin/activate
  pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
  pip install keras
  ```
4. Ignore a billion warnings while keras installation compiles stuff for you.
5. Follow instructions in http://keras.io/backend/ to swith backend in Keras (Or maybe you don't have to?). At the moment it works like this:
   ```
   echo '{"epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow"}' > /.keras/keras.json
   ```
6. Fire up your iPython/whateveritscalled server and start playing with the workbook.
7. Improve the models and earn $$$
