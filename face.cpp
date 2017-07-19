//#include <opencv\cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>



#include <iostream>

using namespace std;
using namespace cv;

String face_cascade = "C:/Users/user/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
String img_name1 = "face.jpg";
String img_name2 = "face_1.jpg";

CascadeClassifier face;

/*int main()       // 얼굴 출력, 자르기
{
	Mat img = imread(img_name);

	if (img.data == NULL) {
		cout << img_name << "이미지 열기 실패" << endl;
		return -1;
	}

	if (!face.load(face_cascade)) {
		cout << "Cascade 파일 열기 실패" << endl;
		return -1;
	}

#pragma region 얼굴 검출
	Mat gray;
	cvtColor(img, gray, CV_RGB2GRAY);

	vector<Rect> face_pos;
	face.detectMultiScale(gray, face_pos, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));

	for (int i = 0; i < (int)face_pos.size(); i++) {
		rectangle(img, face_pos[i], Scalar(0, 255, 0), 2);
	}

#pragma endregion

	Mat subImage = img(face_pos[0]);
	namedWindow("얼굴자르기");
	imshow("얼굴자르기", subImage);
	imwrite("face_1.jpg", subImage);
	
	namedWindow("얼굴 검출");
	imshow("얼굴 검출",img);

	waitKey(0);
	return 0;



}*/

int main()     // 얼굴 비교
{
	Mat img1 = imread(img_name1);
	Mat img2 = imread(img_name2);

	if (img1.data == NULL || img2.data == NULL) {
		cout << "이미지 열기 실패" << endl;
		return -1;
	}

	if (!face.load(face_cascade)) {
		cout << "Cascade 파일 열기 실패" << endl;
		return -1;
	}

	Mat gray1;
	cvtColor(img1, gray1, CV_RGB2GRAY);

	vector<Rect> face_pos1;
	face.detectMultiScale(gray1, face_pos1, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));

	for (int i = 0; i < (int)face_pos1.size(); i++) {
		rectangle(img1, face_pos1[i], Scalar(0, 255, 0), 2);
	}

	Mat subImage1 = img1(face_pos1[0]);
	imwrite("face_comp1.jpg", subImage1);


	Mat gray2;
	cvtColor(img2, gray2, CV_RGB2GRAY);

	vector<Rect> face_pos2;
	face.detectMultiScale(gray2, face_pos2, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));

	rectangle(img2, face_pos2[0], Scalar(0, 255, 0), 2);

	Mat subImage2 = img2(face_pos2[0]);
	imwrite("face_comp2.jpg", subImage2);


	Ptr<BRISK> ptrBrisk = BRISK::create();

	// detecting keypoints
	FeatureDetector detector;
	vector<KeyPoint> keypoints1, keypoints2;
	ptrBrisk->detect(subImage1, keypoints1);
	ptrBrisk->detect(subImage2, keypoints2);

	// computing descriptors
	DescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	ptrBrisk->compute(subImage1, keypoints1, descriptors1);
	ptrBrisk->compute(subImage2, keypoints2, descriptors2);

	// matching descriptors
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	// drawing the results
	namedWindow("matches", 1);
	Mat img_matches;
	drawMatches(subImage1, keypoints1, subImage2, keypoints2, matches, img_matches);
	imshow("matches", img_matches);

	if (matches.size() > 50)
		cout << matches.size() << "개 일치!!" << endl;
	
	waitKey(0);



	return 0;
}