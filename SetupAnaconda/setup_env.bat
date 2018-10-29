:: Run this in the Anaconda Prompt

@echo OFF

if %1.==. goto No1

set CurrentPath=%cd%
set ProjectPath=%1
set EnvironmentPath=%ProjectPath%\Env

call conda create -y -n %EnvironmentPath% python=3.6
call activate %EnvironmentPath%

call pip install tensorflow-gpu keras pandas
call conda install -y pillow=5.0 lxml=4.2 jupyter=1.0 matplotlib=2.2 opencv=3.3 scikit-learn=0.20

call git clone https://github.com/tensorflow/models.git %ProjectPath%\models

set ScriptPath=%~dp0

cd %ProjectPath:~0,2%
cd %ProjectPath%\models\research

%ScriptPath%..\protoc-3.4.0-win32\bin\protoc.exe object_detection/protos/*.proto --python_out=.

cd %CurrentPath:~0,2%
cd %CurrentPath%

GOTO End1

:No1
  echo.
  echo Usage: %~n0 PROJECT_PATH
  echo E.g.: %~n0 c:\DeepIGT
  echo.
  echo Note: If admin access is needed to write the environment path, then make sure to start this Anaconda Prompt in Administrator mode.
  echo.
goto End1

:End1
