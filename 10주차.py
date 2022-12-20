#기계학습응용 10주차 과제
from tensorflow.keras import Model  #케라스에서 모델을 가져옴
from tensorflow import keras  #케라스를 가져옴
from tensorflow.keras.models import Sequential  #순차모델을 가져옴
from tensorflow.keras.layers import Dense, Dropout, Flatten  #층, 드롭아웃을 가져옴
from tensorflow.keras.layers import Conv2D, MaxPooling2D  #CNN 모델, max pooling을 가져옴
from tensorflow.keras.datasets import mnist  #mnist 데이터를 가져옴
import matplotlib.pyplot as plt  #matplotlib를 가져옴
import numpy as np  #넘파이를 가져옴
import random  #랜덤 모듈을 가져옴

def load_data():  #데이터를 로드하는 함수
    # 먼저 MNIST 데이터셋을 로드하겠습니다.
    # 케라스는 `keras.datasets`에 널리 사용하는 데이터셋을 로드하기 위한 함수를 제공합니다.
    # 이 데이터셋은 이미 훈련 세트와 테스트 세트로 나누어져 있습니다.
    # 훈련 세트를 더 나누어 검증 세트를 만드는 것이 좋습니다:

    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()  #훈련데이터, 테스트데이터를 로드
    X_train_full = X_train_full.astype(np.float32)  #훈련 X데이터를 float 형태로 바꿈
    X_test = X_test.astype(np.float32)  #테스트 X데이터를 float 형태로 바꿈
    #print(X_train_full.shape, y_train_full.shape)
    #print(X_test.shape, y_test.shape)
    return X_train_full, y_train_full, X_test, y_test  #훈련데이터, 테스트데이터를 리턴

def data_normalization(X_train_full, X_test):  #데이터 정규화하는 함수
    # 전체 훈련 세트를 검증 세트와 (조금 더 작은) 훈련 세트로 나누어 보죠. 또한 픽셀 강도를 255로 나누어 0~1 범위의 실수로 바꾸겠습니다.

    X_train_full = X_train_full / 255  #훈련 데이터를 255로 나눔 => 모든 데이터가 0~1 사이

    X_test = X_test / 255.  #테스트 데이터를 255로 나눔 => 모든 데이터가 0~1 사이
    train_feature = np.expand_dims(X_train_full, axis=3)  #4차원의 형태로 만들기 위해 축을 추가함
    test_feature = np.expand_dims(X_test, axis=3)  #4차원의 형태로 만들기 위해 축을 추가함

    print(train_feature.shape, train_feature.shape)  #훈련 데이터의 형태 출력 (60000, 28, 28, 1)
    print(test_feature.shape, test_feature.shape)  #테스트 데이터의 형태 출력 (10000, 28, 28, 1)

    return train_feature,  test_feature  #훈련데이터, 테스트데이터를 리턴

def draw_digit(num):  #숫자 출력하는 함수(0과 1)
    for i in num:  #num에 있는 내용을 반복
        for j in i:  #i에 있는 내용을 반복
            if j == 0:  #j가 0인 경우
                print('0', end='')  #0 출력
            else :  #j가 0이 아닌 경우
                print('1', end='')  #1 출력
        print()  #한 줄 엔터

def makemodel(X_train, y_train, X_valid, y_valid, weight_init):  #모델을 만드는 함수
    model = Sequential()  #순차모델 생성
    model.add(Conv2D(32, kernel_size=(3, 3),  input_shape=(28,28,1), activation='relu'))  #convolution, 채널 32개, 필터크기 (3,3), 입력형태 (28,28,1), 활성함수 relu
    model.add(MaxPooling2D(pool_size=2))  #크기를 줄이기 위해 pooling을 사용 => max pooling이기 때문에 특정 영역에서 가장 큰 수로 선택됨
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) #convolution, 채널 64개, 필터크기 (3,3), 활성함수 relu
    model.add(MaxPooling2D(pool_size=2))  #크기를 줄이기 위해 pooling을 사용 => max pooling이기 때문에 특정 영역에서 가장 큰 수로 선택됨
    model.add(Dropout(0.25))  #과잉적합을 막기 위해 25%는 훈련하지 않음
    model.add(Flatten())  #데이터를 펼침
    model.add(Dense(128, activation='relu'))  #은닉층, 출력 128개, 활성함수 relu
    model.add(Dense(10, activation='softmax'))  #출력층, softmax를 이용하여 0~9로 10개의 출력
    model.summary()  #모델 요약
    model.compile(loss='sparse_categorical_crossentropy',  #손실함수
                  optimizer='adam',  #최적화 방법
                  metrics=['accuracy'])  #정확도는 accuracy로



    return model  #모델을 리턴

def plot_history(histories, key='accuracy'):  #정확도 그려주는 함수
    plt.figure(figsize=(16,10))  #그림 크기

    for name, history in histories:  #name: 'baseline', history: 모델 저장한 변수
        val = plt.plot(history.epoch, history.history['val_'+key],  #x는 훈련횟수, y는 모델의 val_accuracy
                       '--', label=name.title()+' Val')  #점선 형태로 그리기, 라벨 이름은 baseline val
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),  #x는 훈련횟수, y는 모델의 accuracy, 색은 val[0]으로부터
                 label=name.title()+' Train')  #라벨 이름은 baseline Train

    plt.xlabel('Epochs')  #x축 이름 설정
    plt.ylabel(key.replace('_',' ').title())  #y축 이름은 key에서 _을 공백으로 치환한, 첫 단어의 첫 글자만 대문자로 사용
    plt.legend()  #범례 표시

    plt.xlim([0,max(history.epoch)])  #x축 범위는 0~반복횟수(훈련횟수)
    plt.show()  #그림 보이기

def draw_prediction(pred, k,X_test,y_test,yhat):  #예측 결과 그림을 그리는 함수
    samples = random.choices(population=pred, k=16)  #pred 데이터로, 16개 랜덤으로 고름

    count = 0  #카운트 변수를 0으로 초기화
    nrows = ncols = 4  #행과 열을 4로 초기화
    plt.figure(figsize=(12,8))  #그림 크기를 설정

    for n in samples:  #샘플에 있는 데이터(랜덤 숫자)로 반복문 실행
        count += 1  #카운트 증가
        plt.subplot(nrows, ncols, count)  #4행 4열에 count번째 그림을 그림
        plt.imshow(X_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')  #테스트 이미지의 n번 데이터를, 흑백이미지로, 정확하지 않은 값은 비슷한(이웃한) 값으로
        tmp = "Label:" + str(y_test[n]) + ", Prediction:" + str(yhat[n])  #라벨은 테스트 n번의 이름, 예측은 yhat n번의 이름
        plt.title(tmp)  #타이틀을 위에 입력한 문자열로 설정

    plt.tight_layout()  #레이아웃을 타이트하게(공백 타이트하게)
    plt.show()  #이미지 출력

def evalmodel(X_test,y_test,model):  #모델 평가하는 함수
    yhat = model.predict(X_test)  #x테스트와 모델을 이용해 y값 예측
    yhat = yhat.argmax(axis=1)  #열에서 가장 큰 값 찾음

    print(yhat.shape)  #yhat의 형태를 출력 (10000,)
    answer_list = []  #정답을 맞춘 데이터를 추가하기 위해 빈 리스트 생성

    for n in range(0, len(y_test)):  #정답을 맞춘 데이터의 수만큼 반복
        if yhat[n] == y_test[n]:  #정답을 맞췄다면
            answer_list.append(n)  #정답 리스트에 추가

    draw_prediction(answer_list, 16,X_test,y_test,yhat)  #정답인 데이터를 매개변수로 한 그림 그리는 함수 호출

    answer_list = []  #틀린 데이터를 추가하기 위해 빈 리스트 생성

    for n in range(0, len(y_test)):  #정답을 틀린 데이터의 수만큼 반복
        if yhat[n] != y_test[n]:  #만약 정답을 맞추지 못했다면
            answer_list.append(n)  #틀린 리스트에 추가

    draw_prediction(answer_list, 16,X_test,y_test,yhat)  #틀린 데이터를 매개변수로 한 그림 그리는 함수 호출

def main():
    X_train, y_train, X_test, y_test = load_data()  #해당 함수를 통해 훈련, 테스트 데이터 반환
    X_train, X_test = data_normalization(X_train,  X_test)  #해당 함수를 통해 정규화시킨 데이터 반환

    model= makemodel(X_train, y_train, X_test, y_test,'glorot_uniform')  #훈련 데이터, 테스트 데이터를 보내고 xavier 초기화 방법으로 모델 생성

    baseline_history = model.fit(X_train,  #x훈련 데이터 사용
                                 y_train,  #y 훈련 데이터 사용
                                 epochs=50,  #50회 반복
                                 batch_size=512,  #batch 크기는 512
                                 validation_data=(X_test, y_test),  #검증 데이터는 테스트 데이터로 사용
                                 verbose=2)  #학습 과정을 한 줄씩 보임(진행상황은 보이지 않음)

    evalmodel(X_test, y_test, model)  #테스트 데이터와 모델을 매개변수로 모델 평가 함수 호출
    plot_history([('baseline', baseline_history)])  #반복 횟수에 대한 정확도 그리는 함수 호출

main()  #실행
