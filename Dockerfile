# Use a multi-stage build to optimize the image size

# Stage 1: Build React frontend
FROM node:14 as frontend-builder

WORKDIR /app/frontend

# Copy frontend code
COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# Install serve globally to serve the React app
RUN npm install -g serve

# Stage 2: Build FastAPI backend
FROM python:3.8-slim as backend-builder

WORKDIR /app/backend

# Copy backend code
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

# Stage 3: Final stage
FROM python:3.8-slim

WORKDIR /app

# Copy built frontend from the first stage
COPY --from=frontend-builder /app/frontend/build /app/frontend/build

# Copy backend from the second stage
COPY --from=backend-builder /app/backend /app/backend

# Expose ports
EXPOSE 7860
EXPOSE 8000

# Start both frontend and backend
CMD ["sh", "-c", "cd /app/backend && uvicorn main:app --host 0.0.0.0 --port 8000 & serve -s /app/frontend/build -l 7860"]