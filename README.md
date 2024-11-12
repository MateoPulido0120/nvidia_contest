# Large Language Robotic Arm Model (LLRM)

## Project Overview

The **Large Language Robotic Arm Model (LLRM)** represents a cutting-edge advancement in robotic interaction powered by artificial intelligence. Developed by **Mateo Pulido**, a Mechatronics Engineer and Generative AI Developer, this model seamlessly bridges the gap between human intent and robotic action by integrating advanced natural language processing (NLP), computer vision, and kinematic algorithms. 

LLRM enables intuitive voice or text-based control over robotic movements, processing natural language commands into precise, real-time actions, and adapting to dynamic environments. This flexibility makes it ideal for applications in manufacturing and automation, where efficiency and adaptability are essential. Designed for the **NVIDIA and LlamaIndex Developer Contest**, LLRM showcases a powerful fusion of AI and robotics for real-world utility and enhanced user engagement.

* [Promotional video](https://www.youtube.com/watch?v=imV_XEKuHHY)
* [Functional video](https://youtu.be/bN-aVxDIASw)


---

## About Me

I’m **Mateo Pulido**, a Mechatronics Engineer and currently a Generative AI Developer. My focus is on developing innovative solutions that leverage artificial intelligence for automation, optimization, and enhanced productivity in robotics and industrial processes. I am passionate about creating AI-driven applications that simplify complex tasks and open new possibilities in industrial automation.

## Project Setup

To set up the environment for LLRM, follow these steps:


## CoppeliaSim Education Installation Guide

### For Linux

1. **Download CoppeliaSim Education**:

   Go to the official [CoppeliaSim website](https://www.coppeliarobotics.com/downloads.html) and select the "Education" version for Linux. Download the compressed file (e.g., `CoppeliaSim_Edu_V4_5_0_Ubuntu20_04.tar.xz`).

2. **Extract the downloaded file**:

   Open a terminal in the location where you downloaded the file and use the following command to extract it:
   ```bash
   tar -xf CoppeliaSim_Edu_V4_5_0_Ubuntu20_04.tar.xz

3. **Navigate to the CoppeliaSim directory**:
    ```bash
   cd CoppeliaSim_Edu_V4_5_0_Ubuntu20_04

4. **Run CoppeliaSim:**:
    ```bash
   ./coppeliaSim.sh
   # Note: If you encounter permission issues, make the file executable:
   chmod +x coppeliaSim.sh
   # (Optional) Create a symbolic link to open CoppeliaSim from any location in the terminal:
   sudo ln -s /path/to/directory/CoppeliaSim_Edu_V4_5_0_Ubuntu20_04/coppeliaSim.sh /usr/local/bin/coppeliasim

### For Windows

1. **Download CoppeliaSim Education**:
    Go to the official CoppeliaSim website and select the "Education" version for Windows. Download the .zip file.

2. **Extract the downloaded file**:
    Right-click on the downloaded .zip file and select Extract All, or use an extraction tool like WinRAR or 7-Zip.

3. **Run CoppeliaSim**:
    Open the extracted folder and double-click on coppeliaSim.exe to start the application.

4. **Open escene "brazo_robot" in CoppeliaSim**:
    Located in nvidia_contest/brazo_robot.ttt

5. **Start simulation**:
    click in ▶ to start simulation

## Run complete proyect

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MateoPulido0120/nvidia_contest.git
   cd nvidia_contest

2. **Install dependencies**:
   ```bash
   pip install requirements.txt

3. **Create secretes.toml file at <.streamlit> folder and add Nvidia API_KEY**:
   ```bash
   Nvidia_key = "Your API_KEY"

4. **Run app**:
   ```bash
   streamlit run app.py


## License and Contact

This code is open-licensed and available for anyone who may find it useful. Feel free to use, modify, and share it in your own projects.

If you have any questions or need further assistance, you can reach me at: [mateo010120@gmail.com](mailto:mateo010120@gmail.com).
