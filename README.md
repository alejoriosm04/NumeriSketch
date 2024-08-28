# NumeriSketch

**NumeriSketch** is a web-based application designed to visualize and graph various numerical methods covered in the course "Numerical Analysis" (ST0256) taught at EAFIT University by Prof. Julian Rendon. This project provides an intuitive and interactive way to explore complex mathematical concepts.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/alejoriosm04/NumeriSketch
   cd numerisketch
    ```

2. **Create virtual environment:**

   ```bash
   python3 -m venv venv
   ```

3. **Activate virtual environment:**

    On Windows:

    ```bash
    venv\Scripts\activate
    ```

    On macOS and Linux:

    ```bash
    source venv/bin/activate
    ```

4. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Setup

1. **Create the database and apply migrations:**

    ```bash
    python manage.py migrate
    ```

2. **Run the development server:**

    ```bash
    python manage.py runserver
    ```

3. **Open your browser and go to `http://http://127.0.0.1:8000/` to access NumeriSketch.