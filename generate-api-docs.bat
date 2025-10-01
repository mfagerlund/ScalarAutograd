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
REM NOTE: This file must be kept in sync with .husky/pre-commit
REM If you update the list of files here, update the pre-commit hook too
type dist\Value.d.ts dist\V.d.ts dist\Optimizers.d.ts dist\Losses.d.ts dist\Vec2.d.ts dist\Vec3.d.ts dist\NonlinearLeastSquares.d.ts dist\LinearSolver.d.ts > etc\api-full.d.ts
if %errorlevel% neq 0 (
    echo Concatenation failed!
    exit /b %errorlevel%
)

echo.
echo Done! API documentation generated in etc/
