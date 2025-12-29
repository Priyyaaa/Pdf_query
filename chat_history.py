"""
Chat History Module
Handles persistent chat history storage
"""
import json
import os
from typing import List, Dict
from datetime import datetime


class ChatHistory:
    """Manage persistent chat history"""
    
    def __init__(self, history_file: str = "chat_history.json"):
        """
        Initialize chat history manager
        
        Args:
            history_file: Path to JSON file storing chat history
        """
        self.history_file = history_file
        self.history: List[Dict] = []
        self.load_history()
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """
        Add a message to chat history
        
        Args:
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata (e.g., timestamp, sources)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.history.append(message)
        self.save_history()
    
    def get_history(self) -> List[Dict]:
        """Get all chat history"""
        return self.history
    
    def get_conversation_context(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation for context"""
        return self.history[-limit:] if limit > 0 else self.history
    
    def clear_history(self):
        """Clear all chat history"""
        self.history = []
        self.save_history()
    
    def save_history(self):
        """Save history to JSON file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving chat history: {str(e)}")
    
    def load_history(self):
        """Load history from JSON file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except Exception as e:
                print(f"Error loading chat history: {str(e)}")
                self.history = []
        else:
            self.history = []


