# **Real-Time Object Detection and Text Recognition with TTS Support**  

This project demonstrates a **real-time object detection and text recognition system** using a webcam. It leverages advanced deep learning models and tools, such as **YOLOS** for object detection and **EasyOCR** for text recognition, alongside **Pyttsx3** for text-to-speech (TTS). Users can seamlessly switch between detection and recognition modes, with detected objects and recognized texts spoken aloud asynchronously.

---

## **Features**  
1. **Real-Time Object Detection:**  
   - Uses `hustvl/yolos-tiny` to detect objects from a live webcam feed.
   - Draws bounding boxes and labels for detected objects.
   - Automatically speaks out the detected objects' names.  

2. **Text Recognition with EasyOCR:**  
   - Extracts and displays text from real-world objects using OCR.
   - Reads out the recognized text aloud via TTS.  

3. **Speech Control:**  
   - Users can skip ongoing TTS at any time.

4. **Mode Switching with Shortcuts:**  
   - `'o'` – Switch to Object Detection Mode  
   - `'r'` – Switch to Text Recognition Mode  
   - `'s'` – Stop Speech  
   - `'q'` – Quit the Program  

---

## **Setup Instructions**  
1. **Clone the Repository:**  
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install Dependencies:**  
   Make sure you have Python installed. Then, install the necessary packages:  
   ```bash
   pip install torch transformers easyocr pyttsx3 opencv-python-headless
   ```

3. **Model Setup:**  
   The YOLOS model and image processor will be automatically downloaded the first time the code runs.

4. **Run the Application:**  
   ```bash
   python main.py
   ```

---

## **Requirements**  
- Python 3.8 or higher  
- Webcam for live video feed  
- **GPU (optional):** The code will automatically switch to GPU if available, enhancing performance.  

---

## **Usage Guide**  
- **Default Mode:** The app starts in **Object Detection Mode**.  
- **Switching Modes:** Press `'o'` for object detection or `'r'` for text recognition.  
- **Skipping Speech:** If TTS is active, press `'s'` to stop it.  
- **Exiting:** Press `'q'` to quit the application.

---

## **Project Structure**  
```
📦project_root
 ┣ 📜main.py           # Main application file
 ┣ 📜README.md         # Project documentation (this file)
 ┗ 📦requirements.txt  # Dependencies list
```

---

## **Future Improvements**  
- Add **multi-language support** for OCR and TTS.  
- Implement **customizable thresholds** for object detection.  
- Save detected objects and recognized texts to a **log file**.  
- Add **voice commands** to switch modes.

---

## **Acknowledgments**  
- [Transformers by Hugging Face](https://huggingface.co/transformers/)  
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)  
- [OpenCV](https://opencv.org/)  

---

## **License**  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.  

---

Feel free to reach out for any questions or contributions! 🚀
