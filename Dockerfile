# Main Dockerfile for MCP Server deployment
# Use the base image with all dependencies pre-installed
FROM mcp-server-base:latest

# Set working directory
WORKDIR /app

# Copy the application code
COPY . .

# Expose the port the MCP server runs on
# Azure Web Apps typically use port 8000 or the PORT environment variable
EXPOSE 8000

# Set environment variables for the application
ENV PYTHONPATH=/app

# Set a default PORT for local testing (Azure will override this)
ENV PORT=8000

# Run the MCP server
# Use exec form to ensure proper signal handling
CMD ["python", "-u", "-m", "mcp_server.main"]
