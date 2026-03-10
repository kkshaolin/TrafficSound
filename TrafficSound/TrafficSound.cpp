#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <sapi.h>
#include <thread> // สำหรับแยก Thread ไม่ให้โปรแกรมค้าง

#pragma comment(lib, "sapi.lib")

using namespace cv;
using namespace std;

// ---------------- Object Structure ----------------
class TrackedSign {
public:
    int id;
    Rect rect;
    Point center;
    int inactive_frames = 0;
    string type; // "Stop" หรือ "Warning"
    Mat crop;    // สำหรับส่งให้ AI
    Scalar color;

	bool hasAnnounced = false; // เพื่อป้องกันการพูดซ้ำๆ

    string aiLabel = "";
    bool aiProcessed = false;

    TrackedSign(int _id, Rect _rect, string _type, Mat _frame) {
        id = _id;
        rect = _rect;
        type = _type;
        center = Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
        color = Scalar(rand() % 255, rand() % 255, rand() % 255);
        updateCrop(_frame);
		hasAnnounced = false;
    }

    void update(Rect newRect, Mat _frame) {
        rect = newRect;
        center = Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
        inactive_frames = 0;
        updateCrop(_frame);
    }

    void updateCrop(Mat _frame) {
        // Safe Crop: ป้องกันการตัดรูปหลุดขอบ
        Rect safeRect = rect & Rect(0, 0, _frame.cols, _frame.rows);
        if (safeRect.area() > 0) {
            resize(_frame(safeRect), crop, Size(128, 128)); // ขนาดมาตรฐานสำหรับ AI
        }
    }
};

// ---------------- Global Variables ----------------
int nextID = 0;
vector<TrackedSign> trackedSigns;
cv::dnn::Net gNet;
vector<string> gLabels;
bool gNetLoaded = false;

// ---------------- Detection Functions ----------------

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

// ---------------- DNN Functions ----------------
bool loadModel(const string& model, const string& labelsPath = "") {
    if (!labelsPath.empty()) {
        ifstream f(labelsPath);
        if (f.is_open()) {
            string line;
            while (getline(f, line)) {
                if (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
                if (!line.empty()) gLabels.push_back(line);
            }
            cout << "Labels loaded: " << gLabels.size() << "\n";
        }
    }

    try {
        gNet = cv::dnn::readNetFromONNX(model);

    }
    catch (const cv::Exception& e) {
        cerr << "Error loading model: " << e.what() << "\n";
        return false;
    }
    gNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    gNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    gNetLoaded = true;
    cout << "Model loaded OK\n";
    return true;
}

string classifyCrop(const Mat& crop) {
    if (!gNetLoaded || crop.empty()) return "No Model";

    // 1. Preprocessing: YOLO-cls ส่วนใหญ่ใช้ 224x224 (หรือตามที่คุณเทรนมา เช่น 640x640)
    // สำคัญ: ตรวจสอบขนาด Size(width, height) ให้ตรงกับที่ตอนเทรน
    Mat blob = cv::dnn::blobFromImage(crop, 1.0 / 255.0, Size(640, 640), Scalar(0, 0, 0), true, false);
    gNet.setInput(blob);

    // 2. Forward Pass: ได้ Output เป็น [1, num_classes]
    Mat prob = gNet.forward();

    // 3. หา Class ที่มีคะแนนสูงสุด
    Point classIdPoint;
    double confidence;
    minMaxLoc(prob.reshape(1, 1), nullptr, &confidence, nullptr, &classIdPoint);
    int classId = classIdPoint.x;

    // 4. กรองค่า Confidence (ถ้าโมเดลมั่นใจน้อยกว่า 50% ให้บอกว่า Unknown)
    if (confidence < 0.5) return "Uncertain (" + to_string((int)(confidence * 100)) + "%)";

    // 5. คืนค่า Label
    string label = (classId < (int)gLabels.size()) ? gLabels[classId] : "ID:" + to_string(classId);

    // คืนค่า Label พร้อมเปอร์เซ็นต์ความมั่นใจ
    char buf[32];
    snprintf(buf, sizeof(buf), " %.0f%%", confidence * 100.0);
    return label + string(buf);
}

string detectInCrop(const Mat& crop) {
    if (!gNetLoaded || crop.empty()) return "Empty";

    // 1. Preprocessing: YOLO มักใช้ 640x640 และ Scale 1/255.0
    // อย่าลืม swapRB: true เพราะ OpenCV อ่านเป็น BGR แต่ Model ส่วนใหญ่เทรนด้วย RGB
    Mat blob = cv::dnn::blobFromImage(crop, 1.0 / 255.0, Size(640, 640), Scalar(0, 0, 0), true, false);
    gNet.setInput(blob);

    // 2. Forward Pass
    Mat output = gNet.forward();

    // 3. Post-processing สำหรับ YOLO (Shape: [1, 84, 8400])
    // เราต้องเปลี่ยนจาก [1, 84, 8400] -> [8400, 84] เพื่อให้ไล่เช็คทีละ 'จุดตรวจจับ' ได้
    Mat rawData = output.reshape(1, output.size[1]); // กลายเป็น [84, 8400]
    transpose(rawData, rawData); // กลายเป็น [8400, 84]

    float maxConf = 0;
    int classId = -1;

    // 4. วนลูปหาแถวที่มีคะแนนสูงสุด (ในภาพที่ Crop มา มักจะมีป้ายเดียวเด่นๆ)
    for (int i = 0; i < rawData.rows; i++) {
        // ใน YOLOv8: col 0-3 คือ bbox, col 4 เป็นต้นไปคือ class scores
        Mat scores = rawData.row(i).colRange(4, rawData.cols);
        double confidence;
        Point classIdPoint;
        minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);

        if (confidence > maxConf) {
            maxConf = (float)confidence;
            classId = classIdPoint.x;
        }
    }

    // 5. คืนค่าผลลัพธ์ (กรองเอาเฉพาะที่มั่นใจมากกว่า 25%)
    float threshold = 0.25;
    if (classId != -1 && maxConf > threshold) {
        string label = (classId < (int)gLabels.size()) ? gLabels[classId] : "ID:" + to_string(classId);

        // แถมเลข Confidence ให้ดูด้วย จะได้รู้ว่าโมเดลมั่นใจแค่ไหน
        char buf[32];
        snprintf(buf, sizeof(buf), " (%.0f%%)", maxConf * 100.0);
        return label + string(buf);
    }

    return "Unknown";
}

//---------------- Voice ----------------
void speak(string text) {
    // แยกเป็น Thread เพื่อไม่ให้ OpenCV ค้างขณะพูด
thread t([text]() {
    ISpVoice* pVoice = NULL;
    if (FAILED(::CoInitialize(NULL))) return;

    HRESULT hr = ::CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void**)&pVoice);
    if (SUCCEEDED(hr)) {
        // แปลง string เป็น wstring สำหรับ SAPI
        wstring stemp = wstring(text.begin(), text.end());
        pVoice->Speak(stemp.c_str(), 0, NULL);
        pVoice->Release();
        pVoice = NULL;
    }
    ::CoUninitialize();
    });
t.detach(); // ปล่อยให้ Thread ทำงานไปอิสระ
}

// ---------------- Main ----------------

int main() {
    VideoCapture cap("test.mp4");
    if (!cap.isOpened()) return -1;

    loadModel("trafic.onnx", "trafic_label.txt");

    Mat frame, hsv, mask, mask1, mask2, mask_red, mask_yellow;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        resize(frame, frame, Size(), 0.5, 0.5);

        // 1. ROI Setup
        Rect myROI(frame.cols / 2, 0, frame.cols / 2, frame.rows);
        Mat cropped = frame(myROI).clone();
        GaussianBlur(cropped, cropped, Size(5, 5), 0);
        cvtColor(cropped, hsv, COLOR_BGR2HSV);
		//imshow("Crop", cropped);

        // 2. Color Masking
        inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);
        inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);
        bitwise_or(mask1, mask2, mask_red);
        inRange(hsv, Scalar(20, 100, 100), Scalar(32, 255, 255), mask_yellow);
        bitwise_or(mask_red, mask_yellow, mask);
        //imshow("Mask", mask);

        //Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        //morphologyEx(mask, mask, MORPH_CLOSE, kernel);
        //morphologyEx(mask, mask, MORPH_OPEN, kernel);

        // 3. Find Contours (Current Blobs)
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        struct Blob { Rect r; string type; };
        vector<Blob> currentBlobs;

        for (const auto& cnt : contours) {
            string type = "";
            if (isStopSign(cnt)) type = "STOP";
            else if (isYellowWarningSign(cnt)) type = "WARNING";

            if (type != "") {
                Rect r = boundingRect(cnt);
                r.x += myROI.x; // ปรับพิกัดกลับไปที่ Full Frame
                currentBlobs.push_back({ r, type });
            }
        }

        // 4. Tracking Logic
        for (auto& ts : trackedSigns) ts.inactive_frames++;

        vector<bool> blobUsed(currentBlobs.size(), false);
        for (auto& ts : trackedSigns) {
            float minDist = 100.0f; // ระยะสูงสุดที่ยอมให้ขยับได้ (พิกเซล)
            int bestIdx = -1;

            for (int i = 0; i < currentBlobs.size(); i++) {
                if (blobUsed[i]) continue;
                Point blobCenter(currentBlobs[i].r.x + currentBlobs[i].r.width / 2,
                    currentBlobs[i].r.y + currentBlobs[i].r.height / 2);
                float dist = norm(ts.center - blobCenter);

                if (dist < minDist) {
                    minDist = dist;
                    bestIdx = i;
                }
            }

            if (bestIdx != -1) {
                ts.update(currentBlobs[bestIdx].r, frame);
                blobUsed[bestIdx] = true;
            }
        }

        // เพิ่มป้ายใหม่ที่ยังไม่มี ID
        for (int i = 0; i < currentBlobs.size(); i++) {
            if (!blobUsed[i]) {
                trackedSigns.push_back(TrackedSign(nextID++, currentBlobs[i].r, currentBlobs[i].type, frame));
            }
        }

        // ลบป้ายที่หายไปนานเกิน 20 เฟรม
        trackedSigns.erase(remove_if(trackedSigns.begin(), trackedSigns.end(),
            [](const TrackedSign& ts) { return ts.inactive_frames > 20; }), trackedSigns.end());

        // 5. Visualize
        //for (const auto& ts : trackedSigns) {
        for (auto& ts : trackedSigns){
            if (ts.inactive_frames < 5) {

                if (!ts.aiProcessed) {

                    Mat aiInput;
                    resize(ts.crop, aiInput, Size(640, 640));
                    ts.aiLabel = classifyCrop(aiInput);
                    //ts.aiLabel = detectInCrop(aiInput);
                    ts.aiProcessed = true;
                }

                rectangle(frame, ts.rect, ts.color, 2);
                string label = "ID:" + to_string(ts.id) + " , CLASS:" + ts.aiLabel;
                putText(frame,label,Point(ts.rect.x, ts.rect.y - 10),FONT_HERSHEY_SIMPLEX,0.6,ts.color,2);

                if (!ts.aiLabel.empty() && !ts.hasAnnounced){
                    //resize(aiInput, aiInput, Size(), 0.5, 0.5);
                    //imshow("AI Input", aiInput);
					string massage = label; // คุณสามารถปรับข้อความที่จะพูดได้ตามต้องการ ใช้ if-else
                    if (ts.aiLabel == "STOP"){
                        massage = "Stop sign detected!";
                    }
					speak(massage);
					ts.hasAnnounced = true; // ป้องกันการพูดซ้ำๆ
                }
            }
        }
        //imshow("Result", frame);
        Mat imgGui = Mat::zeros(Size(1366, 768), CV_8UC3);
        rectangle(imgGui, Point(0, 0), Point(imgGui.cols, 45), Scalar(0, 108, 0), FILLED);
        putText(imgGui, "TrafficSound", Point(10, 30),FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
        int cellWidth = imgGui.cols / 2;
        int cellHeight = (imgGui.rows - 45) / 2;

        Mat frameGui;
		resize(frame, frameGui, Size(cellWidth, cellHeight));
        frameGui.copyTo(imgGui(Rect(0, 45, cellWidth, cellHeight)));

		Mat roiGui;
		resize(cropped, roiGui, Size(cellWidth/2, cellHeight));
		roiGui.copyTo(imgGui(Rect(cellWidth+20, 45, cellWidth/2, cellHeight)));

		Mat maskGui;
		resize(mask, maskGui, Size(cellWidth/2, cellHeight));
		cvtColor(maskGui, maskGui, COLOR_GRAY2BGR);
		maskGui.copyTo(imgGui(Rect(0, 45 + cellHeight, cellWidth/2, cellHeight)));

		Mat aiGui;
        if (!trackedSigns.empty()) {
            Mat lastCrop = trackedSigns.back().crop;
            resize(lastCrop, aiGui, Size(cellWidth/4, cellHeight/2));
            aiGui.copyTo(imgGui(Rect(cellWidth, 45 + cellHeight, cellWidth/4, cellHeight/2)));
		}

		imshow("TrafficSound", imgGui);
        if (waitKey(30) == 27) break;
    }
    return 0;
}