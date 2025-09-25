# Memory Management with mem0

This tour booking agent now includes persistent memory management using [mem0](https://docs.mem0.ai/), enabling the agent to remember user preferences, conversation history, and context across sessions.

## Features

### 🧠 Session Memory
- **User Context**: Remembers property preferences, bedroom requirements, pet ownership, and contact information
- **Conversation History**: Stores conversation exchanges with context and metadata
- **State Persistence**: Saves planner state including booking progress and user preferences
- **Cross-Session Continuity**: Users can continue conversations across different calls/sessions

### 🔧 Implementation Details

#### Memory Service (`app/services/memory_service.py`)
- Wraps mem0 functionality for the tour booking domain
- Handles user identification and session management
- Stores structured data from `PlannerState` as natural language memories
- Provides search and retrieval capabilities

#### Agent Planner Integration (`app/services/agent_planner.py`)
- Loads user context on first conversation turn
- Automatically saves conversation exchanges and state updates
- Pre-populates state with remembered preferences
- Enhanced with session context management

#### Realtime Loop Integration (`app/services/realtime_loop.py`)
- Initializes memory service for each connection
- Sets user/session context using Twilio call IDs
- Enables persistent memory across the conversation flow

## Configuration

### Required Environment Variables
```bash
# Required for mem0 functionality
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4

# Optional - Vector database configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Memory Storage
- **Vector Store**: Uses Qdrant for semantic search (defaults to localhost:6333)
- **LLM**: Uses OpenAI GPT-4 for memory processing and inference
- **Embeddings**: Uses OpenAI text-embedding-3-small for semantic search

## Usage Examples

### Basic Memory Operations
```python
from app.services.memory_service import MemoryService
from app.logging.flight_recorder import FlightRecorder

# Initialize
recorder = FlightRecorder()
memory_service = MemoryService(recorder)

# Store conversation
messages = [
    {"role": "user", "content": "I'm interested in 21 West End"},
    {"role": "assistant", "content": "Great! What size apartment?"}
]
await memory_service.store_conversation_memory(
    messages=messages,
    user_id="user_123",
    session_id="session_456"
)

# Retrieve user context
memories = await memory_service.retrieve_user_context("user_123")
summary = await memory_service.get_user_summary("user_123")
```

### Agent Planner with Memory
```python
# The agent planner automatically uses memory when available
planner = AgentPlanner(dispatcher, recorder, memory_service)
planner.set_session_context("user_123", "session_456")

# On first transcript, user context is loaded automatically
response = await planner.process_transcript("Hi, I'm back!")
# Agent will remember previous preferences and continue the conversation
```

## Memory Types Stored

### User Preferences
- Property interest (21 West End, Hudson 360, Riverview Lofts)
- Apartment size preferences (studio, 1BR, 2BR, etc.)
- Pet ownership status
- Budget ranges and move-in dates
- Tour type preferences (virtual vs in-person)

### Contact Information
- Prospect name, email, and phone number
- Previous booking confirmations
- Communication preferences

### Conversation Context
- Complete conversation history with timestamps
- Booking attempts and outcomes
- Property availability discussions
- Policy and pricing inquiries

## Demo Script

Run the memory demonstration:
```bash
python demo_memory.py
```

This script demonstrates:
1. Storing conversation exchanges
2. Saving planner state as memories
3. Retrieving and searching user context
4. Loading context into agent planner
5. Cross-session continuity

## Benefits

### For Users
- **Seamless Experience**: No need to repeat preferences across calls
- **Faster Service**: Agent immediately knows user context and history
- **Personalized Responses**: Tailored recommendations based on past interactions
- **Continuity**: Can pick up conversations where they left off

### For Agents
- **Enhanced Context**: Full user history available for better assistance
- **Improved Conversion**: Better understanding leads to more successful bookings
- **Reduced Friction**: Less time spent gathering basic information
- **Intelligent Routing**: Can prioritize based on user engagement history

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   WebSocket     │    │  RealtimeLoop    │    │  MemoryService  │
│   Connection    │───▶│                  │───▶│                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  AgentPlanner    │    │     mem0        │
                       │                  │    │   (OpenAI +     │
                       │ + Memory Context │    │    Qdrant)      │
                       └──────────────────┘    └─────────────────┘
```

## Troubleshooting

### Memory Service Not Available
- Ensure `OPENAI_API_KEY` is set in environment
- Check Qdrant is running on specified host/port
- Verify network connectivity to OpenAI API

### Memory Not Persisting
- Check user_id and session_id are being set correctly
- Verify memory service initialization in logs
- Ensure async methods are properly awaited

### Performance Considerations
- Memory operations add ~100-200ms to response time
- Consider implementing memory caching for high-traffic scenarios
- Monitor OpenAI API usage and costs

## Future Enhancements

- **Memory Cleanup**: Automatic cleanup of old/irrelevant memories
- **Memory Analytics**: Insights into user behavior and preferences
- **Multi-Modal Memory**: Support for voice patterns and sentiment
- **Memory Sharing**: Cross-agent memory for enterprise scenarios
