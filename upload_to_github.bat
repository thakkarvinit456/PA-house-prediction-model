@echo off
echo =======================================================
echo   Upload House Price Prediction Project to GitHub
echo =======================================================
echo.
echo Make sure you have created an empty repository on GitHub:
echo https://github.com/new
echo.
set /p REPO_URL="Enter your GitHub Repository URL (e.g., https://github.com/your-username/house-price-prediction.git): "
if "%REPO_URL%"=="" (
    echo Error: No URL provided. Exiting.
    pause
    exit /b 1
)

echo.
echo Linking repository to %REPO_URL%...
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git branch -M main

echo.
echo Pushing project files to GitHub...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =======================================================
    echo   SUCCESS! Your project is now published on GitHub!
    echo =======================================================
) else (
    echo.
    echo If authentication failed, sign in using GitHub CLI or a Personal Access Token.
)
echo.
pause
