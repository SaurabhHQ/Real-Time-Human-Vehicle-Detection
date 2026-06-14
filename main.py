import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = "C:/Users/Saurabh verma/Desktop/python/input.mp4"
cap = cv2.VideoCapture(video_path)

prev_car_x = None

def alert():
    print("🚨 HUMAN + VEHICLE DANGER!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (900, 600))
    results = model(frame, verbose=False)

    persons = []
    vehicles = []

    car_center = None
    car_box = None


    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if name == "person":
                persons.append((x1, y1, x2, y2))

            elif name in ["car", "truck", "bus", "motorbike"]:
                vehicles.append((x1, y1, x2, y2))

                if car_center is None:
                    car_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    car_box = (x1, y1, x2, y2)

   
    car_moving = False

    if car_center and prev_car_x:
        if abs(car_center[0] - prev_car_x) > 3:
            car_moving = True

    if car_center:
        prev_car_x = car_center[0]

    
    danger = False

    if car_box:
        x1, y1, x2, y2 = car_box

        front_x1 = x1 - 80
        front_x2 = x2 + 80
        front_y1 = y1 - 150
        front_y2 = y2

        for px1, py1, px2, py2 in persons:
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2

            if front_x1 < pcx < front_x2 and front_y1 < pcy < front_y2:
                if car_moving:
                    danger = True

    
    if danger:
    
        if car_box:
            x1, y1, x2, y2 = car_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        
        for px1, py1, px2, py2 in persons:
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2

            if car_box:
                fx1, fy1 = car_box[0] - 80, car_box[1] - 150
                fx2, fy2 = car_box[2] + 80, car_box[3]

                if fx1 < pcx < fx2 and fy1 < pcy < fy2:
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)

        cv2.putText(frame, "🚨 DANGER!", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        alert()

    else:
        
        cv2.putText(frame, "SAFE ZONE", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Alert System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()