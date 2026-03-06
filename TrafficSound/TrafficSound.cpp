#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// ฟังก์ชันตรวจจับป้ายหยุด (8 เหลี่ยม สีแดง)
bool isStopSign(const vector<Point>& contour) {
    double area = contourArea(contour);
    if (area < 400) return false;

    vector<Point> approx;
    double perimeter = arcLength(contour, true);
    approxPolyDP(contour, approx, 0.025 * perimeter, true);

    if (approx.size() >= 7 && approx.size() <= 9) {
        if (!isContourConvex(approx)) return false;
        Rect boundingBox = boundingRect(approx);
        double aspectRatio = (double)boundingBox.width / boundingBox.height;
        if (aspectRatio >= 0.75 && aspectRatio <= 1.25) return true;
    }
    return false;
}

// ฟังก์ชันตรวจจับป้ายเตือน (4 เหลี่ยมข้าวหลามตัด สีเหลือง)
bool isYellowWarningSign(const vector<Point>& contour) {
    double area = contourArea(contour);
    if (area < 400) return false;

    vector<Point> approx;
    double perimeter = arcLength(contour, true);
    approxPolyDP(contour, approx, 0.04 * perimeter, true);

    if (approx.size() == 4) {
        if (!isContourConvex(approx)) return false;
        Rect boundingBox = boundingRect(approx);
        double aspectRatio = (double)boundingBox.width / boundingBox.height;
        if (aspectRatio >= 0.75 && aspectRatio <= 1.25) return true;
    }
    return false;
}

int main() {
    VideoCapture cap("test.mp4");
    if (!cap.isOpened()) {
        cout << "Cannot open video file" << endl;
        return -1;
    }

    Mat frame, hsv, mask, mask_red, mask1, mask2, mask_yellow, RoiDetect;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        resize(frame, frame, Size(), 0.5, 0.5);

        // 1. กำหนด ROI ฝั่งขวาของภาพ (สมมติว่าป้ายอยู่ทางขวา)
        Rect myROI(frame.cols / 2, 0, frame.cols / 2, frame.rows);
        Mat cropped = frame(myROI).clone();

        GaussianBlur(cropped, cropped, Size(5, 5), 0);

        // 2. Pre-processing
        cvtColor(cropped, hsv, COLOR_BGR2HSV);

        // 3. ทำ Mask สี
        // สีแดง (2 ช่วง)
        inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);
        inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);
        bitwise_or(mask1, mask2, mask_red);

        // สีเหลือง
        inRange(hsv, Scalar(20, 100, 100), Scalar(32, 255, 255), mask_yellow);

        // รวม Mask ทั้งหมด
        bitwise_or(mask_red, mask_yellow, mask);

        //Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        //// 1. กำจัดจุดเล็กๆ (Noise) ออกไปก่อน
        //morphologyEx(mask, mask, MORPH_OPEN, kernel);
        //// 2. เชื่อมพิกเซลที่เหลือให้เป็นก้อนที่สมบูรณ์
        //morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        // 4. ค้นหา Contours
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (const auto& cnt : contours) {
            bool stop = isStopSign(cnt);
            bool warning = isYellowWarningSign(cnt);

            if (stop || warning) {
                Rect r = boundingRect(cnt);

                // ปรับพิกัดกลับไปยังภาพ Full Frame
                Rect actualRect = r;
                actualRect.x += myROI.x;
                actualRect.y += myROI.y;

                // ป้องกัน Rect หลุดขอบภาพ
                actualRect &= Rect(0, 0, frame.cols, frame.rows);

                if (actualRect.area() > 0) {
                    // --- CROP ROI ---
                    RoiDetect = frame(actualRect).clone();
                    imshow("Detected ROI", RoiDetect); // แสดงภาพที่ครอบมาได้

                    // วาดกรอบและข้อความแยกตามประเภท
                    Scalar color = stop ? Scalar(0, 0, 255) : Scalar(0, 255, 255);

                    rectangle(frame, actualRect, color, 3);
                }
            }
        }

        imshow("Result", frame);
		imshow("Mask", mask);
        if (waitKey(30) == 27) break; // กด ESC เพื่อออก
    }

    cap.release();
    destroyAllWindows();
    return 0;
}