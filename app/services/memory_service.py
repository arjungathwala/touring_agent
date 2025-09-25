"""Memory service using mem0 for persistent session memory management."""

from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import asdict

from mem0 import Memory

from app.logging.flight_recorder import FlightRecorder

logger = logging.getLogger(__name__)


class MemoryService:
    """Memory service for managing conversation context and user preferences using mem0."""
    
    def __init__(self, recorder: FlightRecorder) -> None:
        self.recorder = recorder
        self._memory: Optional[Any] = None  # Can be Memory or MemoryClient
        self._initialize_memory()
    
    def _initialize_memory(self) -> None:
        """Initialize mem0 memory with configuration."""
        try:
            # Check for Mem0 Platform API key first (managed solution)
            mem0_api_key = os.getenv("MEM0_API_KEY")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
            if mem0_api_key:
                # Use Mem0 Platform (managed solution)
                from mem0 import MemoryClient
                self._memory = MemoryClient(api_key=mem0_api_key)
                logger.info("Memory service initialized with Mem0 Platform (managed)")
                self.recorder.log("MEMORY", "initialized", provider="mem0_platform")
                
            elif openai_api_key:
                # Fallback to open source version
                os.environ["OPENAI_API_KEY"] = openai_api_key
                self._memory = Memory()
                logger.info("Memory service initialized with Mem0 Open Source")
                self.recorder.log("MEMORY", "initialized", provider="mem0_oss")
                
            else:
                logger.warning("No MEM0_API_KEY or OPENAI_API_KEY found, memory service disabled")
                self.recorder.log("MEMORY", "disabled", reason="no_api_key")
                
        except Exception as e:
            logger.error("Failed to initialize memory service: %s", e)
            self.recorder.log("MEMORY", "init_error", error=str(e))
            self._memory = None
    
    async def store_conversation_memory(
        self, 
        messages: List[Dict[str, str]], 
        user_id: str, 
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store conversation messages in memory."""
        if not self._memory:
            return False
            
        try:
            with self.recorder.stage("MEMORY", operation="store_conversation"):
                # Check if using platform or open source
                if hasattr(self._memory, 'add') and 'MemoryClient' not in str(type(self._memory)):
                    # Open source version - use messages format
                    result = self._memory.add(
                        messages=messages,
                        user_id=user_id,
                        agent_id="tour-booking-agent",
                        run_id=session_id,
                        metadata={
                            "timestamp": datetime.now().isoformat(),
                            "conversation_type": "tour_booking",
                            **(metadata or {})
                        }
                    )
                else:
                    # Platform version - add each message separately
                    for msg in messages:
                        conversation_text = f"{msg['role']}: {msg['content']}"
                        result = self._memory.add(
                            conversation_text,
                            user_id=user_id,
                            metadata={
                                "agent_id": "tour-booking-agent",
                                "session_id": session_id,
                                "timestamp": datetime.now().isoformat(),
                                "conversation_type": "tour_booking",
                                "message_role": msg['role'],
                                **(metadata or {})
                            }
                        )
                
            logger.info("Stored conversation memory for user %s, session %s", user_id, session_id)
            self.recorder.log("MEMORY", "conversation_stored", user_id=user_id, session_id=session_id)
            return True
            
        except Exception as e:
            logger.error("Failed to store conversation memory: %s", e)
            self.recorder.log("MEMORY", "store_error", error=str(e))
            return False
    
    async def store_planner_state(
        self, 
        state: Any,  # Use Any to avoid circular import, will be PlannerState at runtime
        user_id: str, 
        session_id: str
    ) -> bool:
        """Store planner state as structured memory."""
        if not self._memory:
            return False
            
        try:
            with self.recorder.stage("MEMORY", operation="store_state"):
                # Convert state to memory-friendly format
                state_dict = asdict(state)
                
                # Create memory entries for key preferences
                memory_entries = []
                
                if state.property_id and state.property_name:
                    memory_entries.append(f"User is interested in {state.property_name} (ID: {state.property_id})")
                
                if state.desired_bedrooms is not None:
                    bedroom_text = "studio" if state.desired_bedrooms == 0 else f"{state.desired_bedrooms} bedroom"
                    memory_entries.append(f"User prefers {bedroom_text} apartments")
                
                if state.has_pets:
                    memory_entries.append("User has pets")
                
                if state.tour_type:
                    memory_entries.append(f"User prefers {state.tour_type} tours")
                
                if state.move_in_date:
                    memory_entries.append(f"User wants to move in on {state.move_in_date}")
                
                if state.budget_min or state.budget_max:
                    budget_text = f"${state.budget_min or 0}-${state.budget_max or 'unlimited'}"
                    memory_entries.append(f"User's budget range is {budget_text}")
                
                if state.prospect_name:
                    memory_entries.append(f"User's name is {state.prospect_name}")
                
                if state.prospect_email:
                    memory_entries.append(f"User's email is {state.prospect_email}")
                
                if state.prospect_phone:
                    memory_entries.append(f"User's phone is {state.prospect_phone}")
                
                # Store each memory entry
                for entry in memory_entries:
                    if hasattr(self._memory, 'add') and 'MemoryClient' not in str(type(self._memory)):
                        # Open source version
                        self._memory.add(
                            entry,
                            user_id=user_id,
                            agent_id="tour-booking-agent", 
                            run_id=session_id,
                            metadata={
                                "type": "planner_state",
                                "timestamp": datetime.now().isoformat(),
                                "raw_state": state_dict
                            }
                        )
                    else:
                        # Platform version
                        try:
                            self._memory.add(
                                entry,
                                user_id=user_id,
                                metadata={
                                    "agent_id": "tour-booking-agent",
                                    "session_id": session_id,
                                    "type": "planner_state",
                                    "timestamp": datetime.now().isoformat(),
                                    "raw_state": str(state_dict)  # Convert to string for platform
                                }
                            )
                        except Exception as e:
                            logger.warning("Failed to store individual state entry '%s': %s", entry, e)
                            # Continue with other entries
                
            logger.info("Stored planner state for user %s, session %s", user_id, session_id)
            self.recorder.log("MEMORY", "state_stored", user_id=user_id, session_id=session_id, entries=len(memory_entries))
            return True
            
        except Exception as e:
            logger.error("Failed to store planner state: %s", e)
            self.recorder.log("MEMORY", "state_store_error", error=str(e))
            return False
    
    async def retrieve_user_context(
        self, 
        user_id: str, 
        query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant user context and preferences."""
        if not self._memory:
            return []
            
        try:
            with self.recorder.stage("MEMORY", operation="retrieve_context"):
                if hasattr(self._memory, 'search') and 'MemoryClient' not in str(type(self._memory)):
                    # Open source version
                    if query:
                        memories = self._memory.search(
                            query=query,
                            user_id=user_id,
                            agent_id="tour-booking-agent"
                        )
                    else:
                        memories = self._memory.get_all(
                            user_id=user_id,
                            agent_id="tour-booking-agent"
                        )
                else:
                    # Platform version - use filters
                    if query:
                        memories = self._memory.search(
                            query=query,
                            user_id=user_id
                        )
                    else:
                        memories = self._memory.get_all(
                            user_id=user_id
                        )
                
            # Ensure memories is always a list
            if not isinstance(memories, list):
                if hasattr(memories, '__iter__') and not isinstance(memories, (str, bytes)):
                    memories = list(memories)
                else:
                    memories = [memories] if memories else []
                
            logger.info("Retrieved %d memories for user %s", len(memories), user_id)
            self.recorder.log("MEMORY", "context_retrieved", user_id=user_id, count=len(memories))
            return memories
            
        except Exception as e:
            logger.error("Failed to retrieve user context: %s", e)
            self.recorder.log("MEMORY", "retrieve_error", error=str(e))
            return []
    
    async def retrieve_session_context(
        self, 
        user_id: str, 
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Retrieve context specific to a session."""
        if not self._memory:
            return []
            
        try:
            with self.recorder.stage("MEMORY", operation="retrieve_session"):
                memories = self._memory.get_all(
                    user_id=user_id,
                    agent_id="tour-booking-agent",
                    run_id=session_id
                )
                
            logger.info("Retrieved %d session memories for user %s, session %s", len(memories), user_id, session_id)
            self.recorder.log("MEMORY", "session_retrieved", user_id=user_id, session_id=session_id, count=len(memories))
            return memories
            
        except Exception as e:
            logger.error("Failed to retrieve session context: %s", e)
            self.recorder.log("MEMORY", "session_retrieve_error", error=str(e))
            return []
    
    async def update_user_preference(
        self, 
        memory_id: str, 
        updated_data: str
    ) -> bool:
        """Update a specific user preference."""
        if not self._memory:
            return False
            
        try:
            with self.recorder.stage("MEMORY", operation="update_preference"):
                self._memory.update(memory_id=memory_id, data=updated_data)
                
            logger.info("Updated memory %s", memory_id)
            self.recorder.log("MEMORY", "preference_updated", memory_id=memory_id)
            return True
            
        except Exception as e:
            logger.error("Failed to update preference: %s", e)
            self.recorder.log("MEMORY", "update_error", error=str(e))
            return False
    
    async def get_memory_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get the history of changes for a specific memory."""
        if not self._memory:
            return []
            
        try:
            with self.recorder.stage("MEMORY", operation="get_history"):
                history = self._memory.history(memory_id=memory_id)
                
            logger.info("Retrieved history for memory %s", memory_id)
            self.recorder.log("MEMORY", "history_retrieved", memory_id=memory_id)
            return history
            
        except Exception as e:
            logger.error("Failed to get memory history: %s", e)
            self.recorder.log("MEMORY", "history_error", error=str(e))
            return []
    
    async def delete_user_memories(self, user_id: str) -> bool:
        """Delete all memories for a user."""
        if not self._memory:
            return False
            
        try:
            with self.recorder.stage("MEMORY", operation="delete_user"):
                self._memory.delete_all(user_id=user_id)
                
            logger.info("Deleted all memories for user %s", user_id)
            self.recorder.log("MEMORY", "user_deleted", user_id=user_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete user memories: %s", e)
            self.recorder.log("MEMORY", "delete_error", error=str(e))
            return False
    
    async def get_user_summary(self, user_id: str) -> str:
        """Get a summary of what we know about the user."""
        memories = await self.retrieve_user_context(user_id)
        
        if not memories:
            return "No previous interaction history found."
        
        # Extract key information
        summary_parts = []
        
        for memory in memories[-10:]:  # Last 10 memories for recency
            if isinstance(memory, dict):
                # Handle different memory formats from mem0
                text = memory.get("text") or memory.get("memory") or str(memory)
                if any(keyword in text.lower() for keyword in ["prefers", "interested", "wants", "has", "budget", "name", "email", "phone"]):
                    summary_parts.append(text)
        
        if summary_parts:
            return "Previous context: " + " | ".join(summary_parts[:5])  # Limit to 5 most relevant
        else:
            return "User has previous interactions but no specific preferences captured."
    
    def is_available(self) -> bool:
        """Check if memory service is available."""
        return self._memory is not None
