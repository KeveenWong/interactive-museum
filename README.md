# Interactive Art Gallery with Computer Vision

An immersive, hands-free interactive art gallery experience powered by computer vision and AI. This project demonstrates the potential of accessible technology using only a webcam to create engaging digital art experiences.

## 📷 Screenshots
<img width="420" height="270" alt="image" src="https://github.com/user-attachments/assets/c5a9dcdc-4a3f-40c3-8dc6-3229d46f1309" /> <img width="420" height="270" alt="image" src="https://github.com/user-attachments/assets/1cded4be-1523-4cde-9704-aca069002594" />

<img width="870" height="528" alt="image" src="https://github.com/user-attachments/assets/a91d2182-99ba-4453-b400-2dbbf805fd8d" />

<img width="960" height="500" alt="image" src="https://github.com/user-attachments/assets/d4f1dabe-3832-4863-b631-b8bf23a4ff46" />


## 🎯 Objective

- **Create an immersive, hands-free interactive art gallery experience** using computer vision and AI
- **Support users with limited mobility** or in AR/VR contexts
- **Enable digital artists** to showcase their work in an accessible and engaging manner
- **Explore the current scope of homemade CV solutions** with just a webcam
- **Zero-cost implementation** utilizing robust, real-time pose detection
- **Demonstrate the power** of combining various tools for interactive experiences

## 🛠️ Tech Stack

### Computer Vision & AI
- **OpenCV** - Real-time webcam processing and frame manipulation
- **MediaPipe** - ML-powered facial landmark detection and face mesh mapping
- **OpenAI GPT-4** - AI art guide for contextual artwork information

### 3D Rendering & Game Engine
- **Ursina Engine** - 3D scene rendering, GUI management, and entity interactions
- **Panda3D** - Underlying 3D graphics engine

### Audio & Speech
- **SpeechRecognition** - Voice command processing
- **PyAudio** - Audio input handling

### Additional Libraries
- **NumPy** - Mathematical computations
- **python-dotenv** - Environment variable management

## 🎮 Interactive Controls
- **Head Movement**: Look around the gallery by moving your head
- **Single Blink**: Teleport to highlighted floor tiles or interact with artworks
- **Long Blink (1+ seconds)**: Close artwork information panels
- **Voice Commands**: Say "art guide" to ask questions about displayed artworks

## 📊 Eye Aspect Ratio (EAR) Methodology

The Eye Aspect Ratio is a crucial metric for reliable blink detection, calculated using facial landmarks from MediaPipe. When the EAR drops below a threshold, we detect a blink.

### EAR Formula
```python
def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.7+
- Webcam
- OpenAI API key (for AI art guide feature)

### Installation
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd interactive-museum
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   NOTE: You will need OpenAI credits to call the LLM.
   
4. **Run the application**
   ```bash
   python ursina_scene.py
   ```

---

*This project demonstrates the intersection of computer vision, 3D graphics, and LLMs to create accessible and engaging digital experiences.*
