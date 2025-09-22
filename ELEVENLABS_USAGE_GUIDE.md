# ElevenLabs SDK Usage Guide

## Current Implementation ✅

Your codebase has been updated to use the official ElevenLabs Python SDK correctly. Here's what was changed:

### 1. Dependencies Updated
- Added `elevenlabs` to `requirements.txt`
- Import the official SDK: `from elevenlabs.client import ElevenLabs`

### 2. ElevenLabsClient Refactored
The `ElevenLabsClient` class now:
- Uses the official `ElevenLabs` client instead of manual HTTP calls
- Leverages SDK's `generate()` method with streaming support
- Uses `eleven_turbo_v2_5` model for faster real-time synthesis
- Maintains compatibility with your Twilio μ-law format requirement
- Properly handles errors through the SDK

### 3. Key Improvements
- **Better Performance**: Streaming audio generation reduces latency
- **Proper Error Handling**: SDK handles HTTP errors and retries automatically  
- **Model Selection**: Using the fastest model (`eleven_turbo_v2_5`) for real-time use
- **Voice Settings**: Maintained your stability/similarity_boost settings
- **Format Compatibility**: Still outputs `ulaw_8000` format for Twilio

## Environment Variables Required

```bash
ELEVENLABS_API_KEY=your_api_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Optional, defaults to Rachel
```

## Installation

```bash
pip install elevenlabs
# or
poetry add elevenlabs
```

## Current Usage Pattern

```python
# Your current pattern (now using SDK internally):
elevenlabs_client = ElevenLabsClient(recorder)
audio_bytes = await elevenlabs_client.synthesize("Hello, how can I help you?")
```

## Advanced: ElevenLabs Conversational AI Integration

For even better performance, consider upgrading to ElevenLabs Conversational AI agents:

### Benefits
- **Lower Latency**: Direct voice-to-voice processing (no STT/TTS pipeline)
- **Natural Conversations**: Built-in conversation management
- **Barge-in Support**: Native interruption handling
- **Tool Calling**: Direct function calling integration

### Implementation Example

```python
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

class ElevenLabsConversationalAgent:
    def __init__(self, agent_id: str, api_key: str):
        self.client = ElevenLabs(api_key=api_key)
        self.conversation = Conversation(
            self.client,
            agent_id,
            requires_auth=True,
            audio_interface=DefaultAudioInterface(),
            callback_agent_response=self._on_agent_response,
            callback_user_transcript=self._on_user_transcript,
        )
    
    def _on_agent_response(self, response: str):
        # Handle agent response
        print(f"Agent: {response}")
    
    def _on_user_transcript(self, transcript: str):
        # Handle user input
        print(f"User: {transcript}")
    
    async def start_conversation(self, user_id: str = None):
        self.conversation.start_session(user_id=user_id)
        conversation_id = self.conversation.wait_for_session_end()
        return conversation_id
```

## Migration Path

### Phase 1: Current (Completed ✅)
- Use ElevenLabs SDK for TTS only
- Keep existing Deepgram STT + OpenAI planning architecture

### Phase 2: Hybrid Approach
- Use ElevenLabs for TTS (current)
- Consider ElevenLabs for STT as well
- Keep OpenAI for planning and tool calling

### Phase 3: Full Conversational AI
- Replace entire pipeline with ElevenLabs Conversational AI
- Configure agent with your tools (availability, booking, etc.)
- Simplify architecture significantly

## Testing Your Current Implementation

1. Install the SDK: `pip install elevenlabs`
2. Set your API key: `export ELEVENLABS_API_KEY=your_key`
3. Run your application - the TTS should work identically but with better performance

## Troubleshooting

### Common Issues
1. **API Key**: Ensure `ELEVENLABS_API_KEY` is set
2. **Voice ID**: Use valid voice ID or remove for default
3. **Format**: The SDK should handle `ulaw_8000` format correctly
4. **Quotas**: Monitor your ElevenLabs usage quotas

### Fallback Behavior
Your implementation gracefully falls back to generating a tone if:
- No API key is provided
- ElevenLabs API is unavailable
- Any errors occur during synthesis

This ensures your voice agent continues working even if ElevenLabs is unavailable.
