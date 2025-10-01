"""
LLM Tool Call Logger
This module provides comprehensive logging for all tool calls made during MCP execution.
"""

import json
import datetime
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading

@dataclass
class ToolCall:
    """Represents a single tool call with all relevant information"""
    timestamp: str
    iteration: int
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    execution_time: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class LLMSession:
    """Represents a complete LLM session with all tool calls"""
    session_id: str
    start_time: str
    end_time: Optional[str]
    total_iterations: int
    tool_calls: List[ToolCall]
    final_answer: Optional[str]
    query: str

class LLMLogger:
    """Comprehensive logger for LLM tool calls"""
    
    def __init__(self, log_file: str = "llm_tool_calls.log"):
        self.log_file = log_file
        self.session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.datetime.now().isoformat()
        self.tool_calls: List[ToolCall] = []
        self.current_iteration = 0
        self.lock = threading.Lock()
        
    def log_tool_call(self, tool_name: str, arguments: Dict[str, Any], 
                     result: Any, execution_time: float, success: bool = True, 
                     error_message: str = None):
        """Log a single tool call"""
        with self.lock:
            tool_call = ToolCall(
                timestamp=datetime.datetime.now().isoformat(),
                iteration=self.current_iteration,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                execution_time=execution_time,
                success=success,
                error_message=error_message
            )
            self.tool_calls.append(tool_call)
            
            # Write to log file immediately
            self._write_tool_call_to_file(tool_call)
            
    def set_iteration(self, iteration: int):
        """Set the current iteration number"""
        with self.lock:
            self.current_iteration = iteration
            
    def end_session(self, final_answer: str = None, query: str = None):
        """End the current session and create summary"""
        end_time = datetime.datetime.now().isoformat()
        
        session = LLMSession(
            session_id=self.session_id,
            start_time=self.start_time,
            end_time=end_time,
            total_iterations=self.current_iteration + 1,
            tool_calls=self.tool_calls.copy(),
            final_answer=final_answer,
            query=query
        )
        
        # Write session summary to file
        self._write_session_summary(session)
        return session
        
    def _write_tool_call_to_file(self, tool_call: ToolCall):
        """Write individual tool call to log file"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                log_entry = {
                    "type": "tool_call",
                    "session_id": self.session_id,
                    "data": asdict(tool_call)
                }
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Error writing tool call to log: {e}")
            
    def _write_session_summary(self, session: LLMSession):
        """Write complete session summary to file"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                log_entry = {
                    "type": "session_summary",
                    "data": asdict(session)
                }
                f.write(json.dumps(log_entry, indent=2) + '\n')
        except Exception as e:
            print(f"Error writing session summary to log: {e}")
            
    def get_tool_call_summary(self) -> Dict[str, Any]:
        """Get a summary of all tool calls"""
        with self.lock:
            if not self.tool_calls:
                return {"message": "No tool calls recorded"}
                
            # Count tool calls by name
            tool_counts = {}
            total_execution_time = 0
            successful_calls = 0
            failed_calls = 0
            
            for call in self.tool_calls:
                tool_name = call.tool_name
                if tool_name not in tool_counts:
                    tool_counts[tool_name] = 0
                tool_counts[tool_name] += 1
                
                total_execution_time += call.execution_time
                
                if call.success:
                    successful_calls += 1
                else:
                    failed_calls += 1
                    
            return {
                "session_id": self.session_id,
                "total_tool_calls": len(self.tool_calls),
                "tool_call_counts": tool_counts,
                "total_execution_time": total_execution_time,
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "average_execution_time": total_execution_time / len(self.tool_calls) if self.tool_calls else 0
            }

# Global logger instance
_global_logger = None

def get_logger() -> LLMLogger:
    """Get the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = LLMLogger()
    return _global_logger

def log_tool_call(tool_name: str, arguments: Dict[str, Any], result: Any, 
                 execution_time: float, success: bool = True, error_message: str = None):
    """Convenience function to log a tool call"""
    logger = get_logger()
    logger.log_tool_call(tool_name, arguments, result, execution_time, success, error_message)

def set_iteration(iteration: int):
    """Convenience function to set iteration"""
    logger = get_logger()
    logger.set_iteration(iteration)

def end_session(final_answer: str = None, query: str = None):
    """Convenience function to end session"""
    logger = get_logger()
    return logger.end_session(final_answer, query)

def get_summary():
    """Convenience function to get tool call summary"""
    logger = get_logger()
    return logger.get_tool_call_summary()
