#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {

    VideoCapture cap("test.mp4");
    if (!cap.isOpened()) {
        cout << "Cannot open camera" << endl;
        return -1;
    }

    Mat frame, hsv, mask1, mask2, mask, result;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        resize(frame, frame, Size(), 0.5, 0.5);

        // 1️⃣ แปลงเป็น HSV
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // 2️⃣ กำหนดช่วงสีแดง (ต้องใช้ 2 ช่วง)
        Scalar lower_red1(0, 100, 100);
        Scalar upper_red1(10, 255, 255);

        Scalar lower_red2(160, 100, 100);
        Scalar upper_red2(180, 255, 255);

        inRange(hsv, lower_red1, upper_red1, mask1);
        inRange(hsv, lower_red2, upper_red2, mask2);

        mask = mask1 | mask2;

        // ลด noise
        GaussianBlur(mask, mask, Size(5, 5), 0);
        morphologyEx(mask, mask, MORPH_CLOSE,
            getStructuringElement(MORPH_RECT, Size(5, 5)));

        // 3️⃣ หา Contours
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (size_t i = 0; i < contours.size(); i++) {

            double area = contourArea(contours[i]);

            // กรองขนาดเล็ก
            if (area > 300) {

                // 4️⃣ ตรวจสอบรูปทรง
                vector<Point> approx;
                approxPolyDP(contours[i], approx, 0.02 * arcLength(contours[i], true), true);

                // ถ้าใกล้เคียง 8 เหลี่ยม
                if (approx.size() >= 7 && approx.size() <= 9) {

                    // วาดกรอบ
                    drawContours(frame, contours, i, Scalar(0, 255, 0), 3);

                    Rect bounding = boundingRect(approx);
                    rectangle(frame, bounding, Scalar(255, 0, 0), 2);

                    putText(frame, "STOP SIGN",
                        Point(bounding.x, bounding.y - 10),
                        FONT_HERSHEY_SIMPLEX,
                        0.8,
                        Scalar(0, 0, 255),
                        2);
                }
            }
        }
        
        imshow("Frame", frame);
        imshow("Mask", mask);

        if (waitKey(30) == 27) break; // ESC ออก
    }

    cap.release();
    destroyAllWindows();
    return 0;
}