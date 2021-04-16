set CondaPath=%1
Echo %CondaPath%
set EnvName=%2
Echo %EnvName%
set CurrentPath=%3
Echo %CurrentPath%
set NetworkType=%4
set ModelPath=%5
set ModelName=%6
set OutputType=%7
set IncomingHostName=%8
set IncomingPort=%9
shift
set OutgoingHostName=%9
shift
set OutgoingPort=%9
shift
set DeviceName=%9

call cmd.exe /K %CurrentPath%\StartNeuralNet.bat %CondaPath% %EnvName% %CurrentPath% %NetworkType% %ModelPath% %ModelName% %OutputType% %IncomingHostName% %IncomingPort% %OutgoingHostName% %OutgoingPort% %DeviceName%