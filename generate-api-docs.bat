@echo off
echo Generating API documentation...
echo.

echo [1/2] Building TypeScript...
call npm run build
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)

echo.
echo [2/2] Running API Extractor...
call npx api-extractor run --local
if %errorlevel% neq 0 (
    echo API Extractor failed!
    exit /b %errorlevel%
)

echo.
echo Done! API documentation generated in etc/
