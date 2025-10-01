#!/usr/bin/env python3
"""
Script to run talk2mcp with comprehensive logging and then analyze the results.
"""

import subprocess
import sys
import os
from log_analyzer import LogAnalyzer

def run_with_logging():
    """Run the talk2mcp script with logging enabled"""
    print("Starting talk2mcp with comprehensive logging...")
    print("="*60)
    
    try:
        # Run the logging version
        result = subprocess.run([
            sys.executable, "talk2mcp_with_logging.py"
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print(f"\nReturn code: {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running script: {e}")
        return False

def analyze_logs():
    """Analyze the generated logs"""
    print("\n" + "="*60)
    print("ANALYZING GENERATED LOGS")
    print("="*60)
    
    try:
        analyzer = LogAnalyzer()
        analyzer.print_detailed_report()
        
        # Export to CSV if requested
        export_csv = input("\nExport detailed data to CSV? (y/n): ").lower().strip()
        if export_csv == 'y':
            analyzer.export_to_csv("detailed_tool_calls_analysis.csv")
            print("Data exported to detailed_tool_calls_analysis.csv")
        
        return True
        
    except Exception as e:
        print(f"Error analyzing logs: {e}")
        return False

def main():
    """Main execution function"""
    print("LLM Tool Call Logger and Analyzer")
    print("="*60)
    
    # Check if log file already exists
    if os.path.exists("llm_tool_calls.log"):
        overwrite = input("Log file already exists. Overwrite? (y/n): ").lower().strip()
        if overwrite != 'y':
            print("Using existing log file for analysis...")
            analyze_logs()
            return
    
    # Run the script with logging
    success = run_with_logging()
    
    if success:
        print("\nScript completed successfully!")
        
        # Analyze the logs
        analyze_logs()
    else:
        print("\nScript failed, but checking for partial logs...")
        if os.path.exists("llm_tool_calls.log"):
            analyze_logs()

if __name__ == "__main__":
    main()
