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

4. Follow instructions in http://keras.io/backend/ to swith backend in Keras (Or maybe you don't have to?)
5. Fire up your iPython/whateveritscalled server and start playing with the workbook.
6. Improve the models and earn $$$
