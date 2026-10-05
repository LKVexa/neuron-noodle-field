@echo off
setlocal
cd /d "%~dp0"
if not exist "dotnet\NeuralPlayer\bin\Release\net10.0-windows\NeuralPlayer.exe" goto missing
start "Neuron Noodle Field" "dotnet\NeuralPlayer\bin\Release\net10.0-windows\NeuralPlayer.exe" "%~dp0examples-native\or.tiff"
exit /b
:missing
echo This is source code. Download the ready-to-run Windows ZIP from:
echo https://github.com/LKVexa/neuron-noodle-field/releases/tag/v0.4.0
pause
