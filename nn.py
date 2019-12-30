from tensorflow import keras 
from tensorflow.python.keras import backend as k
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
from confusion import plot_confusion_matrix
import warnings
warnings.filterwarnings("ignore")


def nn(X_train, X_test, y_train, y_test, class_num, input_dim, epochs, batch_size, optimizer, loss):
    # Neural network
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    # model.add(Dense(4, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(class_num, activation='softmax'))

    if optimizer=="sgd":
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        optimizer_using=sgd
    elif optimizer=="adam":
        optimizer_using="adam"
    
    if loss=="binary":
        loss_using='binary_crossentropy'
    elif loss=="categorical":
        loss_using='categorical_crossentropy'
    
    model.compile(loss=loss_using, optimizer=optimizer_using, metrics=['accuracy'])  

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    y_pred = model.predict(X_test)
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
    #Converting one hot encoded test label to label
    test = list()
    for i in range(len(y_test)):
        test.append(np.argmax(y_test[i]))
        
    from sklearn.metrics import accuracy_score
    a = accuracy_score(pred,test)
    print("")
    print('Accuracy is:', a*100)
    print("")
    print("----------------------")

    if class_num==2:
        classes = ['0', '1']
    else:
        classes = ['0', '1', '2', '3']
    np.set_printoptions(precision=2)
    plot_confusion_matrix(test, pred, classes=classes,
                            normalize=False,
                            title=None,
                            cmap=plt.cm.Blues)
    plt.show()


    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=100, batch_size=64)

    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss']) 
    plt.plot(history.history['val_loss']) 
    plt.title('Model loss') 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    plt.show()