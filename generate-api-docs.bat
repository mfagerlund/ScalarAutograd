@echo off
echo Generating API documentation...
echo.

echo [1/3] Building TypeScript...
call npm run build
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)

echo.
echo [2/3] Running API Extractor...
call npx api-extractor run --local
if %errorlevel% neq 0 (
    echo API Extractor failed!
    exit /b %errorlevel%
)

echo.
echo [3/3] Concatenating type definitions...
type dist\Value.d.ts dist\V.d.ts dist\Optimizers.d.ts dist\Losses.d.ts > etc\api-full.d.ts
if %errorlevel% neq 0 (
    echo Concatenation failed!
    exit /b %errorlevel%
)

echo.
echo Done! API documentation generated in etc/
