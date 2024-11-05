# ML_Kit-Model

1. ML Kit 커스텀 모델 구조

**컨볼루션 신경망(CNN)**을 사용하여 이미지로부터 특징을 추출합니다.
눈의 상태(열림, 감김)를 분류하기 위한 특징을 학습합니다.

분류기

추출된 특징을 기반으로 눈 상태를 분류합니다.
**Recurrent Neural Network(RNN)**을 추가로 사용하여 시간적 정보를 활용.

졸음 판단 로직

일정 시간 동안 눈 감김 상태가 지속되면 졸음으로 판단합니다.
EAR(Eye Aspect Ratio) 등의 지표를 사용할 수도 있습니다.

모델 선택
**전이 학습(Transfer Learning)**을 활용하여 사전 학습된 모델(VGG, ResNet 등)을 기반으로 커스텀 모델을 구축할 수 있습니다.
모바일 환경에서는 모델의 경량화가 중요하므로, MobileNet이나 EfficientNet-Lite와 같은 경량 모델을 사용하는 것이 좋습니다.

모델 준비
데이터 수집 및 전처리
다양한 조명, 각도, 표정에서의 얼굴 및 눈 이미지 데이터를 수집합니다.
데이터는 열린 눈과 감긴 눈으로 레이블링되어야 합니다.
이미지 크기 및 형식을 모델에 맞게 전처리합니다.
모델 구축 및 학습
TensorFlow 또는 TensorFlow Keras를 사용하여 모델을 구축합니다.

#예제 코드 (간단한 CNN 모델):

import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # 열림/감김 분류
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))


#모델 변환
#학습된 TensorFlow 모델을 TensorFlow Lite 형식으로 변환
tflite_convert \
  --output_file=model.tflite \
  --saved_model_dir=/path/to/saved_model

#변환 시 최적화 옵션을 사용하여 모델 크기를 줄이고 성능을 향상
tflite_convert \
  --output_file=model.tflite \
  --saved_model_dir=/path/to/saved_model \
  --optimizations=OPTIMIZE_FOR_SIZE


**ML Kit에 모델 통합하기
모델 배포 옵션
앱에 직접 포함: 모델 파일을 앱 패키지에 포함합니다.
원격 모델 배포(Firebase ML): Firebase를 통해 모델을 원격으로 호스팅하고 업데이트할 수 있습니다.
앱에 모델 포함하기
모델 파일 추가

model.tflite 파일을 Android의 assets 폴더 또는 iOS의 리소스 폴더에 추가합니다.
ML Kit 라이브러리 종속성 추가**

Android (build.gradle):

implementation 'com.google.mlkit:mlkit'
pod 'GoogleMLKit/MLKit'


모델 로딩 및 초기화
Android 예제:

import com.google.mlkit.common.model.LocalModel;
import com.google.mlkit.vision.model.CustomObjectDetectorOptions;
import com.google.mlkit.vision.objects.custom.CustomObjectDetector;
import com.google.mlkit.vision.common.InputImage;

// 로컬 모델 정의
LocalModel localModel = new LocalModel.Builder()
    .setAssetFilePath("model.tflite")
    .build();

// 옵션 설정
CustomObjectDetectorOptions options =
    new CustomObjectDetectorOptions.Builder(localModel)
        .setDetectorMode(CustomObjectDetectorOptions.STREAM_MODE)
        .enableClassification()  // 분류 활성화
        .build();

// 모델 초기화
ObjectDetector objectDetector = ObjectDetection.getClient(options);


iOS 예제:

import MLKit

// 로컬 모델 정의
let localModel = LocalModel(path: Bundle.main.path(forResource: "model", ofType: "tflite")!)

// 옵션 설정
let options = CustomObjectDetectorOptions(localModel: localModel)
options.detectorMode = .stream
options.shouldEnableClassification = true

// 모델 초기화
let objectDetector = ObjectDetector.objectDetector(options: options)


5. 실시간 추론 및 최적화
실시간 영상 처리
카메라로부터 프레임을 실시간으로 캡처하고 모델에 입력합니다.
프레임 처리는 별도의 스레드에서 수행하여 UI 스레드를 차단하지 않도록 합니다.
입력 이미지 생성
Android:

InputImage image = InputImage.fromByteBuffer(
    byteBuffer,
    imageWidth,
    imageHeight,
    rotationDegree,
    InputImage.IMAGE_FORMAT_NV21 // 또는 적절한 포맷
);



추론 수행
모델에 이미지를 입력하고 결과를 얻습니다.

objectDetector.process(image)
    .addOnSuccessListener(detections -> {
        // 결과 처리
    })
    .addOnFailureListener(e -> {
        // 오류 처리
    });


