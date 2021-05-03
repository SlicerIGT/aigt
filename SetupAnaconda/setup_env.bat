@echo ON

if %1.==. goto No1

set CurrentPath=%cd%
set EnvironmentPath=%1


:: Create the environment with modules and activate it
:: This environment does not require previous CUDA and CuDNN installations
:: You may optionally upgrade TensorFlow with this command
:: call pip install tensorflow-gpu==2.3.1

call conda create -y -p %EnvironmentPath%
call activate %EnvironmentPath%
call conda install -y tensorflow-gpu=2.1
call conda install -y pandas opencv jupyter scikit-learn scikit-image matplotlib
call pip install girder-client pyigtl

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