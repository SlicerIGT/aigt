
@echo OFF

if %1.==. goto No1

set CurrentPath=%cd%
set EnvironmentPath=%1

:: Create the environment and activate it

call conda create -y -n %EnvironmentPath% tensorflow-gpu
call activate %EnvironmentPath%

:: Install additional packages

call conda install -y keras==2.2.4 pandas==0.24.2 opencv==3.4.2 jupyter git==2.20.1 scikit-learn==0.21.1

:: Install pyIGTLink from source

call git clone -b pyIGTLink_client https://github.com/SlicerIGT/pyIGTLink.git %ProjectPath%\pyIGTLink
call pip install -e %ProjectPath%\pyIGTLink

:: Install keras-vis from source

cd %EnvironmentPath%

call git clone https://github.com/raghakot/keras-vis.git %EnvironmentPath%\keras-vis
cd keras-vis
call python setup.py install

cd %CurrentPath%


:: Exiting install script

GOTO End1

:No1
  echo.
  echo Usage: %~n0 ENVIRONMENT_PATH
  echo E.g.: %~n0 c:\MyProject
  echo.
  echo Note: If admin access is needed to write the environment path, then make sure to start this Anaconda Prompt in Administrator mode.
  echo.
goto End1

:End1
