# Stage 1: Build React frontend
FROM node:14 AS frontend-builder

WORKDIR /app/frontend

# Copy frontend code
COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# Stage 2: Build FastAPI backend
FROM python:3.8-slim AS backend-builder

WORKDIR /app/backend

# Install system dependencies
RUN apt-get update && apt-get install -y gcc libpq-dev

# Copy backend code
COPY backend/requirements.txt ./

# Debugging: List contents to verify requirements.txt is present
RUN ls -la

# Install Python dependencies
RUN pip install --no-cache-dir -r ./requirements.txt && pip freeze

COPY backend/ ./

# Stage 3: Final stage
FROM python:3.8-slim

WORKDIR /app

# Install serve globally to serve the React app
RUN pip install uvicorn fastapi && apt-get update && apt-get install -y nodejs npm && npm install -g serve

# Copy built frontend from the first stage
COPY --from=frontend-builder /app/frontend/build /app/frontend/build

# Copy backend from the second stage
COPY --from=backend-builder /app/backend /app/backend

# Expose ports
EXPOSE 7860
EXPOSE 8000

# Start both frontend and backend
CMD ["sh", "-c", "cd /app/backend && uvicorn main:app --host 0.0.0.0 --port 8000 & serve -s /app/frontend/build -l 7860"]