@echo OFF

if %1.==. goto No1

set CurrentPath=%cd%
set EnvironmentPath=%1


:: Create the environment with modules and activate it

call conda create -y -p %EnvironmentPath% python=3.7 tensorflow pandas opencv jupyter scikit-learn scikit-image matplotlib
call activate %EnvironmentPath%

call pip install girder-client


:: Install pyIGTLink from source

:: call git clone -b pyIGTLink_client https://github.com/SlicerIGT/pyIGTLink.git %EnvironmentPath%\pyIGTLink
:: call pip install -e %EnvironmentPath%\pyIGTLink


:: Install keras-vis from source

:: cd %EnvironmentPath%

:: call git clone https://github.com/raghakot/keras-vis.git %EnvironmentPath%\keras-vis
:: cd keras-vis
:: call python setup.py install

:: cd %CurrentPath%


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