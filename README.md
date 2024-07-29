# 💡 SignItOut
## ASL fingerspelling to text application

Electrical Engineering and CS HUJI 4th year project.

<!-- cool project cover image -->
![Example image Raizel](https://github.com/raizelgestetner/ASL_fingerspelling/blob/main/images/Raizel_spelling.jpg)
![Example image Ilay](https://github.com/raizelgestetner/ASL_fingerspelling/blob/main/images/Ilay_spelling.jpg)

<!-- table of content -->
## Table of Contents
- [The Team](#the-team)
- [Project Description](#project-description)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installing](#installing)
- [Deployment](#deployment)
- [Built With](#built-with)
- [Acknowledgments](#acknowledgments)

## 👥 The Team
**Team Members**
- [Raizel Gestetner](mailto:raizel.gestetner@mail.huji.ac.il)
- [Ilay Chen](mailto:ilay.chen@mail.huji.ac.il)

**Supervisor**
- [Matan Levy](mailto:levy@cs.huji.ac.il)

## 📚 Project Description
This project is an Android application that translates American Sign Language Fingerspelling from real-time video to text.

The main components of the project are:
- 3 mediaPipe neural networks - extract hand, face and post body coordinates
- ASL fingerspelling to text neural network

The main technologies used in this project are:
- Android studio app IDE for Android apps, using Kotlin programming language
- Google Colab for initial running of the modules and other helpful Python scripts
- TensorFlow lite modules

## ⚡ Getting Started
To get started and edit the app, you just need to clone the app which is under the directory [ASLTranslate](https://github.com/raizelgestetner/ASL_fingerspelling/tree/main/ASLTranslate "ASLTranslate").

### 🧱 Prerequisites
- Android Studio IDE
- Android device or emulator

### 🏗️ Installing
Clone the repository:

    git clone https://github.com/raizelgestetner/ASL_fingerspelling/ASLTranslate.git

Open the files in Android Studio, install the packages that are used in the project (for example tflit). The IDE will mark the uninstalled packages. Then you can start editing.

## 🚀 Deployment
You may download the app for usage from Google Play in the following link:
[SignItOut on Google Play](https://play.google.com/store/apps/details?id=com.huji.SignItOut&hl=en)

## ⚙️ Built With
- [ASL fingerspelling to text network](https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434485)
- [mediaPipe body coordinate extracting networks](https://ai.google.dev/edge/mediapipe/solutions/guide)

## 🙏 Acknowledgments
- To the developers of mediaPipe and to Darragh & Dieter, the developers of the ASL module
- To our adviser Matan Levy who helped guide us throughout our journey