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

// �� �̹����� ���絵�� ����ϴ� �Լ�
double calculateImageSimilarity(const Mat& img1, const Mat& img2) {
    Mat hash1, hash2;
    Ptr<img_hash::PHash> hasher = img_hash::PHash::create();
    hasher->compute(img1, hash1);
    hasher->compute(img2, hash2);
    return hasher->compare(hash1, hash2);
}

// Ư¡ ��Ī�� ���� ��� �κ��� ��Ҵ��� �����ϴ� �Լ�
void describeAndShowSimilarRegions(const Mat& img1, const Mat& img2, Mat& outputImg) {
    Ptr<ORB> detector = ORB::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // ��Ī ����� �׷��� outputImg�� ����
    drawMatches(img1, keypoints1, img2, keypoints2, matches, outputImg);
}

// ���� ������ ���� �̹����� ã�� �Լ�
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
    // ���絵�� �ۼ�Ʈ�� ��ȯ (0�� �� 100%, �ִ밪(100)�� �� 0%)
    similarityPercent = 100 - minSimilarity;
    return bestMatch;
}

// ���� �̹����� �ε��ϴ� �Լ�
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
    // ��Ʈ�� ���� ��ķ�� ����Ͽ� ���� ĸó �ʱ�ȭ (ī�޶� ID: 0)
    VideoCapture cap(0);

    // ī�޶� �ػ� ����
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    // ī�޶� �� �� ���� ��� ����
    if (!cap.isOpened()) {
        cout << "ī�޶� �� �� �����ϴ�." << endl;
        return -1;
    }

    // �� ����� �ε�
    CascadeClassifier face_cascade;
    if (!face_cascade.load("C:\\Users\\zkdlf\\Downloads\\haarcascade_frontalface_alt.xml")) { // ��θ� �ùٸ��� ����
        cout << "�� ����� ������ �ε��� �� �����ϴ�." << endl;
        return -1;
    }

    // ���� �̹��� �ε�
    vector<Mat> animalImages;
    vector<string> imagePaths;
    string baseDir = "C:\\Users\\zkdlf\\Downloads\\archive (1)\\afhq\\train";  // ��θ� �ùٸ��� ����
    loadAnimalImages(baseDir, animalImages, imagePaths);

    // ĸó�� �̹��� ������ ���� Mat ����
    Mat img;

    while (true) {
        // ���� ��Ʈ������ �������� ĸó
        cap >> img;

        // �̹��� �¿� ����
        flip(img, img, 1);

        // �׷��̽����� ��ȯ
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);

        // �� ����
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.3, 5); // �Ű����� ����

        // ������ �� ������ �簢�� �׸���
        for (const Rect& face : faces) {
            rectangle(img, face, Scalar(0, 255, 0), 2);

            // ����� �� �κ��� ����
            Mat faceROI = img(face);

            // ���� �̹��� �����ͺ��̽��� ���Ͽ� ���� ������ ���� �̹��� ã��
            Mat bestMatchImage;
            double similarityPercent = 0.0;
            Mat outputImg;
            string bestMatch = findMostSimilarAnimal(faceROI, animalImages, imagePaths, bestMatchImage, similarityPercent, outputImg);
            cout << fixed << setprecision(1); // �Ҽ��� ù° �ڸ����� ����ϵ��� ����
            cout << "Most similar animal image: " << bestMatch << " (" << similarityPercent << "% similar)" << endl;

            // ������ ���� �̹����� ȭ�鿡 ǥ��
            if (!bestMatchImage.empty()) {
                imshow("Best Match Animal", bestMatchImage);
                imshow("Matched Regions", outputImg);
            }
        }

        // ĸó�� �̹����� ȭ�鿡 ǥ��
        imshow("Camera Feed", img);

        // ESC Ű�� ������ ������ �����ϰ� ���α׷��� ����
        if (waitKey(1) == 27) {
            break;
        }
    }

    // ī�޶� ���ҽ� ���� �� ��� â �ݱ�
    cap.release();
    destroyAllWindows();

    return 0;
}