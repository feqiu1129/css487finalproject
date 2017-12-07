@echo off

cls
@pushd %~dp0

setlocal EnableDelayedExpansion

del "%~dp0..\output\*boat*.txt"
del "%~dp0..\output\*graf*.txt"

cd ..\
for %%a in ("%cd%") do set programTitle=%%~na
cd src

set "executable=%programTitle%.exe"

start /b /wait /d "%~dp0..\..\x64\Release\" %executable% testScriptConfig.txt

set "resultFile=%~dp0result.txt"
set "correctResultFileLoc=%~dp0Checked-out_results\"
set "testResultFileLoc=%~dp0..\output\"

set "boatFiles=CHoNI_boat_img1-img2.txt CSIFT_boat_img1-img2.txt CSPIN_boat_img1-img2.txt HoNC_boat_img1-img2.txt HoNC+SIFT_boat_img1-img2.txt HoNC3_boat_img1-img2.txt HoNI_boat_img1-img2.txt HoWH_boat_img1-img2.txt OpponentSIFT_boat_img1-img2.txt RGBSIFT_boat_img1-img2.txt RGBSIFT+HoNC+HoWH+HoNI_boat_img1-img2.txt RGSIFT_boat_img1-img2.txt SIFT_boat_img1-img2.txt SPIN_boat_img1-img2.txt"
set "grafFiles=CHoNI_graf_img1-img2.txt CSIFT_graf_img1-img2.txt CSPIN_graf_img1-img2.txt HoNC_graf_img1-img2.txt HoNC+SIFT_graf_img1-img2.txt HoNC3_graf_img1-img2.txt HoNI_graf_img1-img2.txt HoWH_graf_img1-img2.txt OpponentSIFT_graf_img1-img2.txt RGBSIFT_graf_img1-img2.txt RGBSIFT+HoNC+HoWH+HoNI_graf_img1-img2.txt RGSIFT_graf_img1-img2.txt SIFT_graf_img1-img2.txt SPIN_graf_img1-img2.txt"
set "imagesets=boatFiles grafFiles"

del "%resultFile%"

for %%i in (%imagesets%) do (
	echo %%i >> "%resultFile%" 

	for %%j in (!%%i!) do (
		set "correctResultFile=%correctResultFileLoc%%%j"
		set "testResultFile=%testResultFileLoc%%%j"

		fc "!correctResultFile!" "!testResultFile!" >nul

		if errorlevel 1 (
		    echo %%j files comparison: FAIL >> "%resultFile%"
		) else (
		    echo %%j files comparison: PASS >> "%resultFile%"
		)

	)

	echo( >> "%resultFile%" 

)

find /c "FAIL" "%resultFile%"

echo(

if errorlevel 1 (
	echo The test has PASSED
) else (
	echo The test has FAILED
)

echo(

endlocal

pause