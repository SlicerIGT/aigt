@echo ON

if %1.==. goto No1

set CurrentPath=%cd%
set EnvironmentPath=%1


:: Create the environment with modules and activate it
:: This environment does not require previous CUDA and CuDNN installations
:: Preferably, keep TensorFlow version in synch with 3D Slicer's Python environment

call conda create -y -p %EnvironmentPath%
call activate %EnvironmentPath%
call conda install -y tensorflow-gpu pandas scikit-learn scikit-image matplotlib jupyter opencv
call pip install girder-client pyigtl imutils

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