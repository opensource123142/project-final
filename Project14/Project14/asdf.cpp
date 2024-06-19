#include <opencv2/opencv.hpp>
#include <opencv2/img_hash.hpp>
#include <opencv2/features2d.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <iomanip>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// 두 이미지의 유사도를 계산하는 함수
double calculateImageSimilarity(const Mat& img1, const Mat& img2) {
    Mat hash1, hash2;
    Ptr<img_hash::PHash> hasher = img_hash::PHash::create();
    hasher->compute(img1, hash1);
    hasher->compute(img2, hash2);
    return hasher->compare(hash1, hash2);
}

// 특징 매칭을 통해 어느 부분이 닮았는지 설명하는 함수
void describeAndShowSimilarRegions(const Mat& img1, const Mat& img2, Mat& outputImg) {
    Ptr<ORB> detector = ORB::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // 매칭 결과를 그려서 outputImg에 저장
    drawMatches(img1, keypoints1, img2, keypoints2, matches, outputImg);
}

// 가장 유사한 동물 이미지를 찾는 함수
string findMostSimilarAnimal(const Mat& face, const vector<Mat>& animalImages, const vector<string>& imagePaths, Mat& bestMatchImage, double& similarityPercent, Mat& outputImg) {
    double minSimilarity = std::numeric_limits<double>::max();
    string bestMatch;

    for (size_t i = 0; i < animalImages.size(); ++i) {
        double similarity = calculateImageSimilarity(face, animalImages[i]);
        if (similarity < minSimilarity) {
            minSimilarity = similarity;
            bestMatch = imagePaths[i];
            bestMatchImage = animalImages[i];
            describeAndShowSimilarRegions(face, animalImages[i], outputImg);
        }
    }
    // 유사도를 퍼센트로 변환 (0일 때 100%, 최대값(100)일 때 0%)
    similarityPercent = 100 - minSimilarity;
    return bestMatch;
}

// 동물 이미지를 로드하는 함수
void loadAnimalImages(const string& baseDir, vector<Mat>& animalImages, vector<string>& imagePaths) {
    for (const auto& dirEntry : fs::directory_iterator(baseDir)) {
        if (dirEntry.is_directory()) {
            for (const auto& fileEntry : fs::directory_iterator(dirEntry.path())) {
                if (fileEntry.is_regular_file()) {
                    Mat img = imread(fileEntry.path().string());
                    if (!img.empty()) {
                        animalImages.push_back(img);
                        imagePaths.push_back(fileEntry.path().string());
                    }
                }
            }
        }
    }
}

int main() {
    // 노트북 내장 웹캠을 사용하여 비디오 캡처 초기화 (카메라 ID: 0)
    VideoCapture cap(0);

    // 카메라 해상도 조절
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    // 카메라를 열 수 없는 경우 종료
    if (!cap.isOpened()) {
        cout << "카메라를 열 수 없습니다." << endl;
        return -1;
    }

    // 얼굴 검출기 로드
    CascadeClassifier face_cascade;
    if (!face_cascade.load("C:\\Users\\zkdlf\\Downloads\\haarcascade_frontalface_alt.xml")) { // 경로를 올바르게 설정
        cout << "얼굴 검출기 파일을 로드할 수 없습니다." << endl;
        return -1;
    }

    // 동물 이미지 로드
    vector<Mat> animalImages;
    vector<string> imagePaths;
    string baseDir = "C:\\Users\\zkdlf\\Downloads\\archive (1)\\afhq\\train";  // 경로를 올바르게 설정
    loadAnimalImages(baseDir, animalImages, imagePaths);

    // 캡처된 이미지 저장을 위한 Mat 변수
    Mat img;

    while (true) {
        // 비디오 스트림에서 프레임을 캡처
        cap >> img;

        // 이미지 좌우 반전
        flip(img, img, 1);

        // 그레이스케일 변환
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);

        // 얼굴 검출
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.3, 5); // 매개변수 조정

        // 감지된 얼굴 주위에 사각형 그리기
        for (const Rect& face : faces) {
            rectangle(img, face, Scalar(0, 255, 0), 2);

            // 검출된 얼굴 부분을 추출
            Mat faceROI = img(face);

            // 동물 이미지 데이터베이스와 비교하여 가장 유사한 동물 이미지 찾기
            Mat bestMatchImage;
            double similarityPercent = 0.0;
            Mat outputImg;
            string bestMatch = findMostSimilarAnimal(faceROI, animalImages, imagePaths, bestMatchImage, similarityPercent, outputImg);
            cout << fixed << setprecision(1); // 소수점 첫째 자리까지 출력하도록 설정
            cout << "Most similar animal image: " << bestMatch << " (" << similarityPercent << "% similar)" << endl;

            // 유사한 동물 이미지를 화면에 표시
            if (!bestMatchImage.empty()) {
                imshow("Best Match Animal", bestMatchImage);
                imshow("Matched Regions", outputImg);
            }
        }

        // 캡처한 이미지를 화면에 표시
        imshow("Camera Feed", img);

        // ESC 키를 누르면 루프를 종료하고 프로그램을 종료
        if (waitKey(1) == 27) {
            break;
        }
    }

    // 카메라 리소스 해제 및 모든 창 닫기
    cap.release();
    destroyAllWindows();

    return 0;
}