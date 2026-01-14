
```
# Create a virtual environment
python -m venv vrga_env
vrga_env\Scripts\activate
```

```
install.bat

```

```
# At x64 Native Tools Command Prompt for VS 2019
cd \path\of\the\project
vrga_env\Scripts\activate

git clone https://github.com/ashawkey/diff-gaussian-rasterization.git --recursive
cd diff-gaussian-rasterization
set DISTUTILS_USE_SDK=1
pip install --no-build-isolation .

pip install --no-build-isolation git+https://github.com/camenduru/simple-knn/
```

```
# If gradio version error occured :
pip install --upgrade gradio gradio-client
```

```
pip uninstall -y chumpy
pip install git+https://github.com/elliottzheng/chumpy.git
```


## For Client

Unity Native Web Socket : (Reference)[https://github.com/endel/NativeWebSocket]

Install via UPM (Unity Package Manager)
Open Unity
Open Package Manager Window
Click Add Package From Git URL
Enter URL: https://github.com/endel/NativeWebSocket.git#upm