# Low-Complexity Compressor for 360° Images

This project showcases the spherical image compressor developed for the article "Low-Complexity Compression for 360° Still Images", published in LASCAS 2025 (DOI:_). A more detailed version is available in my [undergraduate thesis](https://lume.ufrgs.br/handle/10183/284279) in Computer Science at the Federal University of Rio Grande do Sul.

## 📁 Project Structure 
	 image_compressor/  
	│── input_images/         	# Folder to store input images  
	│── output_images/        	# Folder to store compressed images  
	│── src/                  	# Main source code of the project  
	│   ├── __init__.py       	# Indicates that src is a Python module  
	│   ├── compressor.py     	# Main image compression class  
	│   ├── image_loader.py   	# Class for loading images  
	│   ├── image_writer.py   	# Class for saving images  
	│   ├── utils.py          	# Auxiliary functions  
	│── tests/                	# Unit tests  
	│   ├── __init__.py       
	│   ├── test_compressor.py  
	│── docs/                 	# Project documentation  
	│── examples/             	# Usage examples  
	│── .gitignore            	# Files to be ignored by Git  
	│── requirements.txt      	# Project dependencies  
	│── setup.bat			  	# Script to set up the environment (Windows) 
	│── setup.sh			  	# Script to set up the environment (Linux/Mac)  
	│── README.md             	# Project information  
	│── main.py              	# Project entry point 

## ⚙️ Environment Setup

### 1️⃣ Technologies
* ![Python](https://img.shields.io/badge/Python-3.13.1-blue)
* ![Pip](https://img.shields.io/badge/Pip-25.0.1-orange)


### 2️⃣ Creation of Virtual Environment and Installation of Dependencies


To set up the development environment, follow the steps below according to your operating system. This will create the virtual environment and install the requirements.

#### **Windows** 🪟
1. Open the terminal in the project directory.
2. Run the setup script to create and activate the virtual environment:
   ```bash
   setup.bat
   ```
3. Activate the virtual environment manually:
	```bash
	.\venv\Scripts\activate
	```
#### **Linux/macOS** 🐧🍏
1. Open the terminal in the project directory.
2. Make the setup script executable:
	```bash
	chmod +x setup.sh
	```
3. Run the setup script to create and activate the virtual environment:
	```bash
	./setup.sh
	```
4. Activate the virtual environment manually:
	```bash
	source venv/bin/activate
	```

## 📕 How to Use
There is a image in `input_images/` that can be used to run the main script:
```bash
python main.py		# On Windows
```
Or
```bash
python3 main.py 	# On Linux/macOS
```

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
