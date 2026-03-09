//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <vector>
//
//using namespace cv;
//using namespace std;
//
//// ---------------- Object Structure ----------------
//class TrackedSign {
//public:
//    int id;
//    Rect rect;
//    Point center;
//    int inactive_frames = 0;
//    string type; // "Stop" หรือ "Warning"
//    Mat crop;    // สำหรับส่งให้ AI
//    Scalar color;
//
//    TrackedSign(int _id, Rect _rect, string _type, Mat _frame) {
//        id = _id;
//        rect = _rect;
//        type = _type;
//        center = Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
//        color = Scalar(rand() % 255, rand() % 255, rand() % 255);
//        updateCrop(_frame);
//    }
//
//    void update(Rect newRect, Mat _frame) {
//        rect = newRect;
//        center = Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
//        inactive_frames = 0;
//        updateCrop(_frame);
//    }
//
//    void updateCrop(Mat _frame) {
//        // Safe Crop: ป้องกันการตัดรูปหลุดขอบ
//        Rect safeRect = rect & Rect(0, 0, _frame.cols, _frame.rows);
//        if (safeRect.area() > 0) {
//            resize(_frame(safeRect), crop, Size(128, 128)); // ขนาดมาตรฐานสำหรับ AI
//        }
//    }
//};
//
//// ---------------- Global Variables ----------------
//int nextID = 0;
//vector<TrackedSign> trackedSigns;
//
//// ---------------- Detection Functions ----------------
//
//// ฟังก์ชันตรวจจับป้ายหยุด (8 เหลี่ยม สีแดง)
//
//bool isStopSign(const vector<Point>& contour) {
//    double area = contourArea(contour);
//    if (area < 400) return false;
//    vector<Point> approx;
//    double perimeter = arcLength(contour, true);
//    approxPolyDP(contour, approx, 0.025 * perimeter, true);
//
//    if (approx.size() >= 7 && approx.size() <= 9) {
//        if (!isContourConvex(approx)) return false;
//        Rect boundingBox = boundingRect(approx);
//        double aspectRatio = (double)boundingBox.width / boundingBox.height;
//        if (aspectRatio >= 0.75 && aspectRatio <= 1.25) return true;
//    }
//    return false;
//}
//
//// ฟังก์ชันตรวจจับป้ายเตือน (4 เหลี่ยมข้าวหลามตัด สีเหลือง)
//bool isYellowWarningSign(const vector<Point>& contour) {
//    double area = contourArea(contour);
//    if (area < 400) return false;
//    vector<Point> approx;
//    double perimeter = arcLength(contour, true);
//    approxPolyDP(contour, approx, 0.04 * perimeter, true);
//
//    if (approx.size() == 4) {
//        if (!isContourConvex(approx)) return false;
//        Rect boundingBox = boundingRect(approx);
//        double aspectRatio = (double)boundingBox.width / boundingBox.height;
//        if (aspectRatio >= 0.75 && aspectRatio <= 1.25) return true;
//    }
//    return false;
//}
//
//// ---------------- Main ----------------
//
//int main_track() {
//    VideoCapture cap("test.mp4");
//    if (!cap.isOpened()) return -1;
//
//    Mat frame, hsv, mask, mask1, mask2, mask_red, mask_yellow;
//
//    while (true) {
//        cap >> frame;
//        if (frame.empty()) break;
//        resize(frame, frame, Size(), 0.5, 0.5);
//
//        // 1. ROI Setup
//        Rect myROI(frame.cols / 2, 0, frame.cols / 2, frame.rows);
//        Mat cropped = frame(myROI).clone();
//        GaussianBlur(cropped, cropped, Size(5, 5), 0);
//        cvtColor(cropped, hsv, COLOR_BGR2HSV);
//
//        // 2. Color Masking
//        inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);
//        inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);
//        bitwise_or(mask1, mask2, mask_red);
//        inRange(hsv, Scalar(20, 100, 100), Scalar(32, 255, 255), mask_yellow);
//        bitwise_or(mask_red, mask_yellow, mask);
//        imshow("Mask", mask);
//
//        //		Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
//        //		morphologyEx(mask, mask, MORPH_CLOSE, kernel);
//        //		morphologyEx(mask, mask, MORPH_OPEN, kernel);
//
//                // 3. Find Contours (Current Blobs)
//        vector<vector<Point>> contours;
//        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//
//        struct Blob { Rect r; string type; };
//        vector<Blob> currentBlobs;
//
//        for (const auto& cnt : contours) {
//            string type = "";
//            if (isStopSign(cnt)) type = "STOP";
//            else if (isYellowWarningSign(cnt)) type = "WARNING";
//
//            if (type != "") {
//                Rect r = boundingRect(cnt);
//                r.x += myROI.x; // ปรับพิกัดกลับไปที่ Full Frame
//                currentBlobs.push_back({ r, type });
//            }
//        }
//
//        // 4. Tracking Logic
//        for (auto& ts : trackedSigns) ts.inactive_frames++;
//
//        vector<bool> blobUsed(currentBlobs.size(), false);
//        for (auto& ts : trackedSigns) {
//            float minDist = 100.0f; // ระยะสูงสุดที่ยอมให้ขยับได้ (พิกเซล)
//            int bestIdx = -1;
//
//            for (int i = 0; i < currentBlobs.size(); i++) {
//                if (blobUsed[i]) continue;
//                Point blobCenter(currentBlobs[i].r.x + currentBlobs[i].r.width / 2,
//                    currentBlobs[i].r.y + currentBlobs[i].r.height / 2);
//                float dist = norm(ts.center - blobCenter);
//
//                if (dist < minDist) {
//                    minDist = dist;
//                    bestIdx = i;
//                }
//            }
//
//            if (bestIdx != -1) {
//                ts.update(currentBlobs[bestIdx].r, frame);
//                blobUsed[bestIdx] = true;
//            }
//        }
//
//        // เพิ่มป้ายใหม่ที่ยังไม่มี ID
//        for (int i = 0; i < currentBlobs.size(); i++) {
//            if (!blobUsed[i]) {
//                trackedSigns.push_back(TrackedSign(nextID++, currentBlobs[i].r, currentBlobs[i].type, frame));
//            }
//        }
//
//        // ลบป้ายที่หายไปนานเกิน 20 เฟรม
//        trackedSigns.erase(remove_if(trackedSigns.begin(), trackedSigns.end(),
//            [](const TrackedSign& ts) { return ts.inactive_frames > 20; }), trackedSigns.end());
//
//        // 5. Visualize
//        for (const auto& ts : trackedSigns) {
//            if (ts.inactive_frames < 5) { // แสดงเฉพาะป้ายที่เพิ่งเห็น (กันการสั่น)
//                rectangle(frame, ts.rect, ts.color, 2);
//                string label = "ID:" + to_string(ts.id) + " " + ts.type;
//                putText(frame, label, Point(ts.rect.x, ts.rect.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, ts.color, 2);
//
//                //                if (!ts.crop.empty()) {
//                //                    imshow("AI Input (ID " + to_string(ts.id) + ")", ts.crop);
//                //                }
//            }
//        }
//
//        imshow("Result", frame);
//        if (waitKey(30) == 27) break;
//    }
//    return 0;
//}