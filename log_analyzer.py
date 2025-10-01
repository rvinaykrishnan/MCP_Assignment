"""
LLM Log Analyzer
This module provides tools to analyze and display LLM tool call logs.
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from llm_logger import ToolCall, LLMSession

class LogAnalyzer:
    """Analyzes LLM tool call logs and provides various views of the data"""
    
    def __init__(self, log_file: str = "llm_tool_calls.log"):
        self.log_file = log_file
        self.tool_calls: List[ToolCall] = []
        self.sessions: List[LLMSession] = []
        self.load_logs()
    
    def load_logs(self):
        """Load all logs from the log file"""
        if not os.path.exists(self.log_file):
            print(f"Log file {self.log_file} not found.")
            return
            
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        entry = json.loads(line)
                        if entry.get("type") == "tool_call":
                            tool_call_data = entry.get("data", {})
                            tool_call = ToolCall(
                                timestamp=tool_call_data.get("timestamp", ""),
                                iteration=tool_call_data.get("iteration", 0),
                                tool_name=tool_call_data.get("tool_name", ""),
                                arguments=tool_call_data.get("arguments", {}),
                                result=tool_call_data.get("result"),
                                execution_time=tool_call_data.get("execution_time", 0.0),
                                success=tool_call_data.get("success", True),
                                error_message=tool_call_data.get("error_message")
                            )
                            self.tool_calls.append(tool_call)
                            
                        elif entry.get("type") == "session_summary":
                            session_data = entry.get("data", {})
                            session = LLMSession(
                                session_id=session_data.get("session_id", ""),
                                start_time=session_data.get("start_time", ""),
                                end_time=session_data.get("end_time"),
                                total_iterations=session_data.get("total_iterations", 0),
                                tool_calls=[],  # Will be populated separately
                                final_answer=session_data.get("final_answer"),
                                query=session_data.get("query", "")
                            )
                            self.sessions.append(session)
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing log line: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error loading logs: {e}")
    
    def get_tool_call_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of all tool calls"""
        if not self.tool_calls:
            return {"message": "No tool calls found in log"}
        
        # Count tool calls by name
        tool_counts = {}
        total_execution_time = 0
        successful_calls = 0
        failed_calls = 0
        iteration_counts = {}
        
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
                
            # Count by iteration
            iteration = call.iteration
            if iteration not in iteration_counts:
                iteration_counts[iteration] = 0
            iteration_counts[iteration] += 1
        
        return {
            "total_tool_calls": len(self.tool_calls),
            "tool_call_counts": tool_counts,
            "total_execution_time": total_execution_time,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "average_execution_time": total_execution_time / len(self.tool_calls) if self.tool_calls else 0,
            "iteration_counts": iteration_counts,
            "unique_tools_used": len(tool_counts)
        }
    
    def get_iteration_breakdown(self) -> Dict[int, List[ToolCall]]:
        """Get tool calls grouped by iteration"""
        iteration_breakdown = {}
        for call in self.tool_calls:
            iteration = call.iteration
            if iteration not in iteration_breakdown:
                iteration_breakdown[iteration] = []
            iteration_breakdown[iteration].append(call)
        return iteration_breakdown
    
    def get_tool_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for each tool"""
        tool_performance = {}
        
        for call in self.tool_calls:
            tool_name = call.tool_name
            if tool_name not in tool_performance:
                tool_performance[tool_name] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "total_execution_time": 0.0,
                    "min_execution_time": float('inf'),
                    "max_execution_time": 0.0,
                    "average_execution_time": 0.0
                }
            
            perf = tool_performance[tool_name]
            perf["total_calls"] += 1
            perf["total_execution_time"] += call.execution_time
            
            if call.execution_time < perf["min_execution_time"]:
                perf["min_execution_time"] = call.execution_time
            if call.execution_time > perf["max_execution_time"]:
                perf["max_execution_time"] = call.execution_time
            
            if call.success:
                perf["successful_calls"] += 1
            else:
                perf["failed_calls"] += 1
        
        # Calculate averages
        for tool_name, perf in tool_performance.items():
            if perf["total_calls"] > 0:
                perf["average_execution_time"] = perf["total_execution_time"] / perf["total_calls"]
            if perf["min_execution_time"] == float('inf'):
                perf["min_execution_time"] = 0.0
        
        return tool_performance
    
    def get_session_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all sessions"""
        session_summaries = []
        
        for session in self.sessions:
            session_summary = {
                "session_id": session.session_id,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "total_iterations": session.total_iterations,
                "final_answer": session.final_answer,
                "query": session.query,
                "duration": None
            }
            
            if session.start_time and session.end_time:
                try:
                    start_dt = datetime.fromisoformat(session.start_time)
                    end_dt = datetime.fromisoformat(session.end_time)
                    session_summary["duration"] = (end_dt - start_dt).total_seconds()
                except:
                    pass
            
            session_summaries.append(session_summary)
        
        return session_summaries
    
    def print_detailed_report(self):
        """Print a detailed analysis report"""
        print("="*80)
        print("LLM TOOL CALL ANALYSIS REPORT")
        print("="*80)
        
        # Overall summary
        summary = self.get_tool_call_summary()
        print(f"\nOVERALL SUMMARY:")
        print(f"  Total Tool Calls: {summary['total_tool_calls']}")
        print(f"  Unique Tools Used: {summary['unique_tools_used']}")
        print(f"  Successful Calls: {summary['successful_calls']}")
        print(f"  Failed Calls: {summary['failed_calls']}")
        print(f"  Total Execution Time: {summary['total_execution_time']:.2f} seconds")
        print(f"  Average Execution Time: {summary['average_execution_time']:.2f} seconds")
        
        # Tool call counts
        print(f"\nTOOL CALL COUNTS:")
        for tool_name, count in summary['tool_call_counts'].items():
            print(f"  {tool_name}: {count}")
        
        # Iteration breakdown
        print(f"\nITERATION BREAKDOWN:")
        iteration_breakdown = self.get_iteration_breakdown()
        for iteration in sorted(iteration_breakdown.keys()):
            calls = iteration_breakdown[iteration]
            print(f"  Iteration {iteration}: {len(calls)} tool calls")
            for call in calls:
                status = "✓" if call.success else "✗"
                print(f"    {status} {call.tool_name}({call.arguments}) -> {call.result}")
                if not call.success and call.error_message:
                    print(f"      Error: {call.error_message}")
        
        # Tool performance
        print(f"\nTOOL PERFORMANCE:")
        tool_performance = self.get_tool_performance()
        for tool_name, perf in tool_performance.items():
            print(f"  {tool_name}:")
            print(f"    Total Calls: {perf['total_calls']}")
            print(f"    Success Rate: {perf['successful_calls']}/{perf['total_calls']} ({perf['successful_calls']/perf['total_calls']*100:.1f}%)")
            print(f"    Avg Execution Time: {perf['average_execution_time']:.3f}s")
            print(f"    Min/Max Execution Time: {perf['min_execution_time']:.3f}s / {perf['max_execution_time']:.3f}s")
        
        # Session summaries
        print(f"\nSESSION SUMMARIES:")
        session_summaries = self.get_session_summary()
        for session in session_summaries:
            print(f"  Session {session['session_id']}:")
            print(f"    Query: {session['query']}")
            print(f"    Iterations: {session['total_iterations']}")
            print(f"    Final Answer: {session['final_answer']}")
            if session['duration']:
                print(f"    Duration: {session['duration']:.2f} seconds")
    
    def export_to_csv(self, output_file: str = "tool_calls_analysis.csv"):
        """Export tool call data to CSV"""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'iteration', 'tool_name', 'arguments', 'result', 
                          'execution_time', 'success', 'error_message']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for call in self.tool_calls:
                writer.writerow({
                    'timestamp': call.timestamp,
                    'iteration': call.iteration,
                    'tool_name': call.tool_name,
                    'arguments': str(call.arguments),
                    'result': str(call.result),
                    'execution_time': call.execution_time,
                    'success': call.success,
                    'error_message': call.error_message or ''
                })
        
        print(f"Tool call data exported to {output_file}")

def main():
    """Main function to run the log analyzer"""
    analyzer = LogAnalyzer()
    analyzer.print_detailed_report()
    
    # Ask if user wants to export to CSV
    export_csv = input("\nExport data to CSV? (y/n): ").lower().strip()
    if export_csv == 'y':
        analyzer.export_to_csv()

if __name__ == "__main__":
    main()
