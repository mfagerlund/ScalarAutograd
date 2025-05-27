#!/bin/sh

set -e

echo "\n--- STARTING NPM RELEASE ---\n"

echo "Step 1: Update version (patch/minor/major):"
read -p "Enter version update type (patch/minor/major): " VTYPE
npm version $VTYPE

echo "\nStep 2: Running tests..."
npm test

echo "\nStep 3: Building project..."
npm run build

echo "\nStep 4: Generating docs..."
npm run docs

echo "\nStep 5: NPM Login (skip if already logged in)"
npm login || true

echo "\nStep 6: Publishing to npm..."
npm publish

echo "\nStep 7: Pushing commits and tags to git..."
git push
git push --tags

echo "\n--- NPM RELEASE COMPLETE ---\n"