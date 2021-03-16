#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <math.h>

using namespace cv;
using namespace std;


// ---------------------------------------------------------------------------
const int POINTS_NUM = 68;

// load face detection and pose estimation models.
dlib::frontal_face_detector dlibSvmFaceDetector;
dlib::shape_predictor dlibSpFaceLandmark;

std::vector<dlib::rectangle> dlibRectsFaces;
std::vector<dlib::full_object_detection> dlibDetsShapes;

// ---------------------------------------------------------------------------

cv::Vec<unsigned char, 3> BilinearInsert(Mat src, int ux, int uy){
    int w = src.cols;
    int h = src.rows;
    int c = src.channels();
    if (c==3){
        int x1 = int(ux);
        int x2 = x1 + 1;
        int y1 = int(uy);
        int y2 = y1 + 1;

        cv::Vec<unsigned char, 3> part1 = src.at<Vec3b>(y1, x1)*(float(x2)-ux)*(float(y2)-uy);
        cv::Vec<unsigned char, 3> part2 = src.at<Vec3b>(y1, x2)*(ux-float(x1))*(float(y2)-uy);
        cv::Vec<unsigned char, 3> part3 = src.at<Vec3b>(y2, x1)*(float(x2)-ux)*(uy-float(y1));
        cv::Vec<unsigned char, 3> part4 = src.at<Vec3b>(y2, x2)*(ux-float(x1))*(uy-float(y1));

        cv::Vec<unsigned char, 3> insertValue = part1+part2+part3+part4;
        return insertValue;
    }
}

//get landmarks
std::vector<dlib::full_object_detection> landmark_dec_dlib_fun(Mat img_src){
    std::vector<dlib::rectangle> dlibRectsFaces;
    dlibSvmFaceDetector = dlib::get_frontal_face_detector();
    // Grab and process frames until the main window is closed by the user.

    cv::Mat cvImgFrameGray;
    cv::cvtColor(img_src, cvImgFrameGray, cv::COLOR_BGR2GRAY);
    dlib::cv_image<unsigned char> dlibImgFrameGray(cvImgFrameGray);
    // Detect faces
    dlibRectsFaces = dlibSvmFaceDetector(dlibImgFrameGray);
    std::vector<dlib::full_object_detection> dlibDetsShapes;
    // Find the landmarks of each face.
    for (unsigned int idxFace = 0; idxFace < dlibRectsFaces.size();
         idxFace++)
    {
        dlibDetsShapes.push_back(dlibSpFaceLandmark(dlibImgFrameGray,
                                                    dlibRectsFaces[idxFace]));
    }
    return dlibDetsShapes;
}

//shift
Mat localTranslationWarp(Mat srcImg, float startX, float startY, float endX, float endY, float radius, float change){
    float ddradius = float(radius*radius);
    Mat copyImg = Mat::zeros(srcImg.rows, srcImg.cols, CV_8UC3);
    srcImg.copyTo(copyImg);
    float ddmc = (endX-startX)*(endX-startX) + (endY-startY)*(endY-startY);
    int H = srcImg.rows;
    int W = srcImg.cols;
    int C = srcImg.channels();
    for (int i = 0; i < W; i++){
        for (int j = 0; j < H; j++){
            if(fabs(i-startX)>radius and fabs(j-startY)>radius) continue;
            float distance = (i-startX)*(i-startX)+(j-startY)*(j-startY);
            if(distance < ddradius){
                float ratio = (ddradius-distance)/(ddradius-distance+ddmc);
                ratio = ratio*ratio;

                int UX = i-change*ratio*(endX-startX);
                int UY = j-change*ratio*(endY-startY);

                cv::Vec<unsigned char, 3> value = BilinearInsert(srcImg, UX, UY);
                copyImg.at<Vec3b>(j, i) = value;
            }
        }
    }
    return copyImg;
}

Mat faceLift(Mat src, float change){
    std::vector<dlib::full_object_detection> dlibDetsShapes;
    dlibDetsShapes = landmark_dec_dlib_fun(src);
    Mat thin_image = Mat::zeros(src.rows, src.cols, CV_8UC3);
    if(dlibDetsShapes.empty()){
        return src;
    }
    for(int i=0; i<dlibDetsShapes.size(); i++){
        Point2f left_landmark = Point2f(dlibDetsShapes[i].part(3).x(), dlibDetsShapes[i].part(3).y());
        Point2f left_landmark_down = Point2f(dlibDetsShapes[i].part(5).x(), dlibDetsShapes[i].part(5).y());
        Point2f right_landmark = Point2f(dlibDetsShapes[i].part(13).x(), dlibDetsShapes[i].part(13).y());
        Point2f right_landmark_down = Point2f(dlibDetsShapes[i].part(15).x(), dlibDetsShapes[i].part(15).y());
        Point2f endPt = Point2f(dlibDetsShapes[i].part(30).x(), dlibDetsShapes[i].part(30).y());

        // 4-6 points
        float r_left = sqrt((left_landmark.x-left_landmark_down.x)*(left_landmark.x-left_landmark_down.x)+
                            (left_landmark.y - left_landmark_down.y) * (left_landmark.y - left_landmark_down.y));

        // 14-16 points
        float r_right=sqrt((right_landmark.x-right_landmark_down.x)*(right_landmark.x-right_landmark_down.x)+
                           (right_landmark.y -right_landmark_down.y) * (right_landmark.y -right_landmark_down.y));
        //
        //瘦左边脸
        thin_image = localTranslationWarp(src,left_landmark.x,left_landmark.y,endPt.x,endPt.y,r_left, change);
        //瘦右边脸
        thin_image = localTranslationWarp(thin_image, right_landmark.x, right_landmark.y, endPt.x,endPt.y, r_right, change);
    }

    return thin_image;
}

//eye
Mat localTranslationWarp_eyeScale(Mat srcImg, float startX, float startY, float endX, float endY, float radius, float change){
    float ddradius = float(radius*radius);
    Mat copyImg = Mat::zeros(srcImg.rows, srcImg.cols, CV_8UC3);
    srcImg.copyTo(copyImg);
    int H = srcImg.rows;
    int W = srcImg.cols;
    int C = srcImg.channels();
    float PtCenterX = 0;
    float PtCenterY = 0;
    PtCenterX = (startX+endX)/2;
    PtCenterY = (startY+endY)/2;
    for (int i = 0; i < W; i++){
        for (int j = 0; j < H; j++){
            int offsetX = i - PtCenterX;
            int offsetY = j - PtCenterY;
            if(fabs(i-PtCenterX)>radius and fabs(j-PtCenterY)>radius) continue;
            float distance = (i-PtCenterX)*(i-PtCenterX)+(j-PtCenterY)*(j-PtCenterY);
            if(distance < ddradius){

                float ScaleFactor = 1 - distance/ddradius;
                ScaleFactor = 1 - change / 500 * ScaleFactor;
                int UX = PtCenterX + offsetX * ScaleFactor;
                int UY = PtCenterY + offsetY * ScaleFactor;

                cv::Vec<unsigned char, 3> value = BilinearInsert(srcImg, UX, UY);
                copyImg.at<Vec3b>(j, i) = value;
            }
        }
    }
    return copyImg;
}


Mat eyeScale(Mat src, float change){
    Mat thin_image = Mat::zeros(src.rows, src.cols, CV_8UC3);
    if(dlibDetsShapes.empty()){
        return src;
    }

    for(int i=0; i<dlibDetsShapes.size(); i++){
        Point2f left_landmark = Point2f(dlibDetsShapes[i].part(36).x(), dlibDetsShapes[i].part(36).y());
        Point2f left_landmark_down = Point2f(dlibDetsShapes[i].part(39).x(), dlibDetsShapes[i].part(39).y());
        Point2f right_landmark = Point2f(dlibDetsShapes[i].part(42).x(), dlibDetsShapes[i].part(42).y());
        Point2f right_landmark_down = Point2f(dlibDetsShapes[i].part(45).x(), dlibDetsShapes[i].part(45).y());

        //radius
        float r_left = sqrt((left_landmark.x-left_landmark_down.x)*(left_landmark.x-left_landmark_down.x)+
                            (left_landmark.y - left_landmark_down.y) * (left_landmark.y - left_landmark_down.y))/2;
        float r_right = sqrt((right_landmark.x-right_landmark_down.x)*(right_landmark.x-right_landmark_down.x)+
                            (right_landmark.y - right_landmark_down.y) * (right_landmark.y - right_landmark_down.y))/2;
        //eyes
        thin_image = localTranslationWarp_eyeScale(src,left_landmark.x,left_landmark.y,left_landmark_down.x,left_landmark_down.y,r_left, change);
        thin_image = localTranslationWarp_eyeScale(thin_image,right_landmark.x,right_landmark.y,right_landmark_down.x,right_landmark_down.y,r_right, change);
    }
    return thin_image;
}

int main(int argc, char** argv)
{
        char *filename = "test.jpg";
        if(argc == 2){
            filename = argv[1];
        }
        float change = 1.0;
        if(argc == 3){
            filename = argv[1];
            change = atof(argv[2]);
        }

        cv::Mat cvImgFrame;
        cv::Mat cvImgFrameGray;

        dlibSvmFaceDetector = dlib::get_frontal_face_detector();
        dlib::deserialize("./data/shape_predictor_68_face_landmarks.dat") >> dlibSpFaceLandmark;

        // Grab and process frames until the main window is closed by the user.

        cvImgFrame = imread(filename, 3);
        cv::namedWindow("src", cv::WINDOW_AUTOSIZE);
        cv::imshow("src", cvImgFrame);

        dlibDetsShapes = landmark_dec_dlib_fun(cvImgFrame);

        clock_t clkBegin;
        clock_t clkEnd;
        clkBegin = clock();
        Mat rstImg = eyeScale(cvImgFrame, change);
        clkEnd = clock();
        cout << "time:" << clkEnd-clkBegin << endl;

        cv::namedWindow("rst", cv::WINDOW_AUTOSIZE);
        cv::imshow("rst", rstImg);
        cv::waitKey(0);

}