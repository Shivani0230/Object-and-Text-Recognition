from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import cv2
import pyttsx3  # For TTS
import easyocr  # For OCR
import threading  # To run TTS asynchronously

# Initialize YOLOS model and image processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny').to(device)
image_processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny')

# Initialize TTS engine
tts = pyttsx3.init()
tts.setProperty('rate', 150)  

# Initialize OCR reader
reader = easyocr.Reader(['en'])  

# Access the webcam
cap = cv2.VideoCapture(0)

# State variables for mode toggling and speech control
mode = "object_detection"  # Default mode
tts_thread = None  # To manage TTS in a separate thread
stop_tts = False  # Flag to stop speech

# Function to run TTS asynchronously
def speak(text):
    global tts_thread, stop_tts
    if tts_thread and tts_thread.is_alive():
        return  
    stop_tts = False  # Reset the stop flag
    tts_thread = threading.Thread(target=lambda: (tts.say(text), tts.runAndWait()))
    tts_thread.start()

# Function to stop ongoing TTS
def stop_speech():
    global stop_tts
    stop_tts = True
    tts.stop()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if mode == "object_detection":
        # Object detection mode
        inputs = image_processor(images=frame_pil, return_tensors="pt").to(device)
        outputs = model(**inputs)

        # Extract detection results
        results = image_processor.post_process_object_detection(
            outputs, threshold=0.9, 
            target_sizes=torch.tensor([frame_pil.size[::-1]]).to(device)
        )[0]

        # Draw bounding boxes and store detected objects
        detected_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i) for i in box.tolist()]
            object_name = model.config.id2label[label.item()]
            detected_objects.append(object_name)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            text = f"{object_name}: {round(score.item(), 2)}"
            cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Speak out detected objects asynchronously
        if detected_objects:
            speak(", ".join(detected_objects))

    elif mode == "text_recognition":
        # Text recognition mode
        ocr_results = reader.readtext(frame)
        recognized_text = " ".join([res[1] for res in ocr_results])

        if recognized_text:
            print(f"Recognized Text: {recognized_text}")
            # Display recognized text on the frame
            cv2.putText(frame, recognized_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Speak the recognized text
            speak(recognized_text)

    # Add labels for shortcuts on the frame
    cv2.putText(frame, "Shortcuts:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "'o' - Object Detection", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "'r' - Text Recognition", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "'s' - Skip Speech", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "'q' - Quit", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow('Real-Time Detection and Recognition', frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('o'):
        mode = "object_detection"
        print("Switched to Object Detection Mode")
    elif key == ord('r'):
        mode = "text_recognition"
        print("Switched to Text Recognition Mode")
    elif key == ord('s'):
        stop_speech()  # Stop ongoing speech
        print("Speech skipped")
    elif key == ord('q'):
        break  # Exit on 'q' key

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
