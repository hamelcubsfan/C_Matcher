#!/bin/bash
set -e

# Check for .env file
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file with your GEMINI_API_KEY."
    echo "Example:"
    echo "GEMINI_API_KEY=your_api_key_here"
    exit 1
fi

echo "Building frontend..."
cd apps/web

# Check if node_modules exists, install if not
if [ ! -d "node_modules" ]; then
    npm install
fi
npm run build

# Go back to root
cd ../..

# Prepare static files for Docker volume mount
echo "Copying static files..."
rm -rf static
# Next.js output is usually in 'out' for static exports, or '.next'
# Assuming 'out' based on typical static site setup, but checking if it exists
if [ -d "apps/web/out" ]; then
    cp -r apps/web/out static
elif [ -d "apps/web/.next" ]; then
    # If not using 'output: export', we might need to copy .next, but the python app likely expects static HTML/JS
    # Let's assume standard static export for now or create a dummy if failed
    echo "Warning: 'out' directory not found. Copying .next (might not work if app expects static HTML)"
    cp -r apps/web/.next static
else
    echo "Error: Frontend build failed to produce 'out' directory."
    exit 1
fi

echo "Starting Docker containers..."
docker compose up --build
