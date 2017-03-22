# CharRecognition2017

Project by: Tarek Khellaf

This "Intelligent Character Recognition" is the final project for the seminar "Artificial Intelligence and Cultural Heritage" (2016/17) at the University of Cologne.

It is planned to create a program which is capable of recognizing Japanese characters (both Kana and Kanji) in documents.
This will be achieved by creating a Convolutional Neural Network (CNN), using training sets of the "ETL Character Database" (http://etlcdb.db.aist.go.jp/), Python and Keras.

It is required to install Keras + TensorFlow backend to run the training (Theano backend requires a few minor changes) and to download the ETL-dataset (currently ETL8G).

Achievenemts so far:
- Load Hiragana from the ETL Character Database (Kanji + Katakana can be added easily)
- Train the model with a decent accuracy
- Recognize character (images of single characters as an input required)
- Implementation of a "Sliding-Window"
- Detection of characters (still not flawless, due to the structure of Japanese characters)

Still to do:
- Change the training set from ETL8 to ETL2, as it contains non-handwritten characters
- Send segmentated characters to the model, return the prediction
- Tune network to increase a higher accuracy
- Convert PDFs to images for further processing

Stay tuned for further information.
