#使用Keras解决线性回归问题
##1.使用Keras的一般步骤
1. 准备数据
1. 搭建网络模型
1. 编译模型
1. 训练
1. 测试
##2.代码
    #coding=utf-8
	from keras.datasets import mnist
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation
	from keras.utils import np_utils
	from keras.optimizers import RMSprop
	import matplotlib.pyplot as plt
	
	classNumber = 10
	batchSize = 64
	epochNum = 10
	
	#准备数据
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
	X_train = X_train.reshape(60000, 784)
	X_test = X_test.reshape(10000, 784)
	X_train = X_train/255
	X_test = X_test/255
	Y_train = np_utils.to_categorical(Y_train, classNumber)
	Y_test = np_utils.to_categorical(Y_test, classNumber)
	
	#创建模型
	model = Sequential()
	model.add(Dense(512, input_shape=(784,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	
	#打印模型概况
	model.summary()
	
	#编译
	model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
	
	#训练，返回history，记录了损失函数及其它指标在训练过程中的变化
	history = model.fit(X_train, Y_train, batch_size=batchSize, nb_epoch=epochNum, verbose=1,
	                    validation_data=(X_test, Y_test))
	
	#绘制准确率曲线
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	
	#测试
	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

输出：

![](https://i.imgur.com/m9dJn7X.jpg)

![](https://i.imgur.com/MmuIyNV.jpg)
