# GitHub Copilot Instructions for Chapter Generation Experiment

## Project Overview

This is an experimental project focused on optimizing the extraction of information chunks (chapters) from video transcripts and associated frames. The goal is to build a robust vector database for accurate query retrieval with minimal latency.

### Key Objectives
- Process video transcripts (5-10 mins to 1-2 hours duration)
- Extract meaningful chapters using LLMs
- Analyze associated video frames
- Create optimized vector embeddings for retrieval
- Focus on chapter generation only (query pipeline is out of scope)

## Coding Guidelines

### General Rules
- **No test files**: Do not create test files, test suites, or testing code unless explicitly requested
- Prioritize experimental and iterative approaches
- Focus on performance optimization and efficiency
- Keep code modular for easy experimentation with different approaches

### LLM Integration
- Use best practices for prompt engineering
- Implement proper error handling for LLM API calls
- Consider token limits and context windows
- Optimize for cost-effectiveness in API usage

### Vector Database
- Focus on embedding quality over quantity
- Implement chunking strategies that preserve semantic meaning
- Consider metadata extraction for better retrieval
- Design for scalability with long-form content

### Video Processing
- Handle variable-length transcripts efficiently
- Process frames at strategic intervals
- Implement transcript-frame alignment logic
- Consider memory optimization for long videos

### Code Structure
- Keep chapter extraction logic separate from embedding generation
- Create reusable components for different extraction strategies
- Document experimental results and findings inline
- Use clear variable names that reflect the experimental nature

## What NOT to do
- Do not create unit tests, integration tests, or test fixtures unless asked
- Do not generate query/retrieval pipeline code
- Do not add production-level error handling unless necessary for experiments
- Avoid over-engineering; prioritize quick iteration
