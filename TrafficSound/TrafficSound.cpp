#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

bool isStopSign(const vector<Point>& contour) {
    // 1. กรองพื้นที่ขั้นต่ำก่อน เพื่อตัด Noise เล็กๆ ออก
    double area = contourArea(contour);
    if (area < 400) return false;

    // 2. ทำ Approximation (ลดทอนจุดให้กลายเป็นรูปทรงเรขาคณิต)
    vector<Point> approx;
    double perimeter = arcLength(contour, true);
    // ค่า epsilon (0.02 - 0.04) ยิ่งมากยิ่งลบรายละเอียดจุดย่อยๆ ออก
    approxPolyDP(contour, approx, 0.025 * perimeter, true);

    // 3. ตรวจสอบจำนวนเหลี่ยม (ป้าย Stop คือ 8 เหลี่ยม)
    // อนุโลมให้ 7-9 เหลี่ยมในกรณีภาพไม่ชัด
    if (approx.size() >= 7 && approx.size() <= 9) {

        // 4. ตรวจสอบว่าเป็นรูปทรง "นูน" หรือไม่ (ป้ายจราจรต้องไม่มีส่วนเว้าเข้า)
        if (!isContourConvex(approx)) return false;

        // 5. ตรวจสอบ Aspect Ratio (ป้าย Stop จะมีความกว้างและสูงใกล้เคียงกัน)
        Rect boundingBox = boundingRect(approx);
        double aspectRatio = (double)boundingBox.width / boundingBox.height;

        // อัตราส่วนควรอยู่ระหว่าง 0.8 ถึง 1.2 (คือเกือบเป็นสี่เหลี่ยมจัตุรัส)
        if (aspectRatio >= 0.75 && aspectRatio <= 1.25) {
            return true;
        }
    }

    return false;
}

int main() {

    VideoCapture cap("test.mp4");
    if (!cap.isOpened()) {
        cout << "Cannot open camera" << endl;
        return -1;
    }

    Mat frame, hsv, mask, mask_red, mask1, mask2, result;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        resize(frame, frame, Size(), 0.5, 0.5);

        // 1. กำหนด ROI และ Crop
        Rect myROI(frame.cols / 2, 0, frame.cols / 2, frame.rows);
        Mat cropped = frame(myROI).clone();

        // 2. Pre-processing บนภาพที่ Crop
        //GaussianBlur(cropped, cropped, Size(5, 5), 0);
        cvtColor(cropped, hsv, COLOR_BGR2HSV);

        // 3. Mask สีแดง
        inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);
        inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);
        bitwise_or(mask1, mask2, mask_red);

        mask = mask_red;

        // 4. ลบ Noise (สำคัญมากเพื่อให้รูปทรงชัดเจนขึ้น)
//        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
//        morphologyEx(mask, mask, MORPH_OPEN, kernel);
//        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        // 5. หา Contours
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (const auto& cnt : contours) {
            if (isStopSign(cnt)) {
                Rect r = boundingRect(cnt);

                // ปรับพิกัดให้ตรงกับ frame หลัก
                Rect actualRect = r;
                actualRect.x += myROI.x;
                actualRect.y += myROI.y;

                rectangle(frame, actualRect, Scalar(0, 255, 0), 3);
                string label = "STOP [" + to_string(actualRect.width) + "x" + to_string(actualRect.height) + "]";
                putText(frame, label, Point(actualRect.x, actualRect.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            }
        }

        imshow("Processed Mask", mask); // ดูว่า Noise หมดไหม
        imshow("Result", frame);

        if (waitKey(30) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}