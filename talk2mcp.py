# talk2mcp.py
import os
import time
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio
from google import genai
from concurrent.futures import TimeoutError
from functools import partial
from llm_logger import get_logger, log_tool_call, set_iteration, end_session, get_summary

# Load environment variables from .env file
load_dotenv()

# Access your API key and initialize Gemini client correctly
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

max_iterations = 3
last_response = None
iteration = 0
iteration_response = []

async def generate_with_timeout(client, prompt, timeout=10):
    """Generate content with a timeout"""
    print("Starting LLM generation...")
    start_time = time.time()
    try:
        # Convert the synchronous generate_content call to run in a thread
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None, 
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            ),
            timeout=timeout
        )
        execution_time = time.time() - start_time
        print("LLM generation completed")
        
        # Log the LLM generation
        log_tool_call(
            tool_name="llm_generation",
            arguments={"model": "gemini-2.0-flash", "prompt_length": len(prompt)},
            result={"response_length": len(response.text) if hasattr(response, 'text') else 0},
            execution_time=execution_time,
            success=True
        )
        
        return response
    except TimeoutError:
        execution_time = time.time() - start_time
        print("LLM generation timed out!")
        log_tool_call(
            tool_name="llm_generation",
            arguments={"model": "gemini-2.0-flash", "prompt_length": len(prompt)},
            result=None,
            execution_time=execution_time,
            success=False,
            error_message="Timeout"
        )
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"Error in LLM generation: {e}")
        log_tool_call(
            tool_name="llm_generation",
            arguments={"model": "gemini-2.0-flash", "prompt_length": len(prompt)},
            result=None,
            execution_time=execution_time,
            success=False,
            error_message=str(e)
        )
        raise

def reset_state():
    """Reset all global variables to their initial state"""
    global last_response, iteration, iteration_response
    last_response = None
    iteration = 0
    iteration_response = []

async def main():
    reset_state()  # Reset at the start of main
    print("Starting main execution...")
    
    # Initialize logger
    logger = get_logger()
    logger.log_file = "demo_llm_tool_calls.log"  # Use demo file directly
    query = """Find the ASCII values of characters in INDIA and then return sum of exponentials of those values."""
    
    try:
        # Create a single MCP server connection
        print("Establishing connection to MCP server...")
        server_params = StdioServerParameters(
            command="python3",
            args=["example2.py"]
        )

        async with stdio_client(server_params) as (read, write):
            print("Connection established, creating session...")
            async with ClientSession(read, write) as session:
                print("Session created, initializing...")
                await session.initialize()
                
                # Get available tools
                print("Requesting tool list...")
                tools_result = await session.list_tools()
                tools = tools_result.tools
                print(f"Successfully retrieved {len(tools)} tools")
                
                # Log the tool discovery
                log_tool_call(
                    tool_name="list_tools",
                    arguments={},
                    result={"tool_count": len(tools), "tool_names": [t.name for t in tools]},
                    execution_time=0.0,
                    success=True
                )
                

                # Create system prompt with available tools
                print("Creating system prompt...")
                print(f"Number of tools: {len(tools)}")
                
                try:
                    # First, let's inspect what a tool object looks like
                    # if tools:
                    #     print(f"First tool properties: {dir(tools[0])}")
                    #     print(f"First tool example: {tools[0]}")
                    
                    tools_description = []
                    for i, tool in enumerate(tools):
                        try:
                            # Get tool properties
                            params = tool.inputSchema
                            desc = getattr(tool, 'description', 'No description available')
                            name = getattr(tool, 'name', f'tool_{i}')
                            
                            # Format the input schema in a more readable way
                            if 'properties' in params:
                                param_details = []
                                for param_name, param_info in params['properties'].items():
                                    param_type = param_info.get('type', 'unknown')
                                    param_details.append(f"{param_name}: {param_type}")
                                params_str = ', '.join(param_details)
                            else:
                                params_str = 'no parameters'

                            tool_desc = f"{i+1}. {name}({params_str}) - {desc}"
                            tools_description.append(tool_desc)
                            print(f"Added description for tool: {tool_desc}")
                        except Exception as e:
                            print(f"Error processing tool {i}: {e}")
                            tools_description.append(f"{i+1}. Error processing tool")
                    
                    tools_description = "\n".join(tools_description)
                    print("Successfully created tools description")
                except Exception as e:
                    print(f"Error creating tools description: {e}")
                    tools_description = "Error loading tools"
                
                print("Created system prompt...")
                
                system_prompt = f"""You are a math agent solving problems in iterations. You have access to various mathematical tools.

Available tools:
{tools_description}

You must respond with EXACTLY ONE line in one of these formats (no additional text):
1. For function calls:
   FUNCTION_CALL: function_name|param1|param2|...
   
2. For final answers:
   FINAL_ANSWER: [number]

Important:
- When a function returns multiple values, you need to process all of them
- Only give FINAL_ANSWER when you have completed all necessary calculations
- Do not repeat function calls with the same parameters

Examples:
- FUNCTION_CALL: add|5|3
- FUNCTION_CALL: strings_to_chars_to_int|INDIA
- FINAL_ANSWER: [42]

DO NOT include any explanations or additional text.
Your entire response should be a single line starting with either FUNCTION_CALL: or FINAL_ANSWER:"""

                query = """Find the ASCII values of characters in INDIA and then return sum of exponentials of those values. """
                print("Starting iteration loop...")
                
                # Use global iteration variables
                global iteration, last_response
                
                while iteration < max_iterations:
                    print(f"\n--- Iteration {iteration + 1} ---")
                    set_iteration(iteration)
                    if last_response is None:
                        current_query = query
                    else:
                        current_query = current_query + "\n\n" + " ".join(iteration_response)
                        current_query = current_query + "  What should I do next?"

                    # Get model's response with timeout
                    print("Preparing to generate LLM response...")
                    prompt = f"{system_prompt}\n\nQuery: {current_query}"
                    try:
                        response = await generate_with_timeout(client, prompt)
                        response_text = response.text.strip()
                        print(f"LLM Response: {response_text}")
                        
                        # Find the FUNCTION_CALL line in the response
                        for line in response_text.split('\n'):
                            line = line.strip()
                            if line.startswith("FUNCTION_CALL:"):
                                response_text = line
                                break
                        
                    except Exception as e:
                        print(f"Failed to get LLM response: {e}")
                        break


                    if response_text.startswith("FUNCTION_CALL:"):
                        _, function_info = response_text.split(":", 1)
                        parts = [p.strip() for p in function_info.split("|")]
                        func_name, params = parts[0], parts[1:]
                        
                        print(f"\nDEBUG: Raw function info: {function_info}")
                        print(f"DEBUG: Split parts: {parts}")
                        print(f"DEBUG: Function name: {func_name}")
                        print(f"DEBUG: Raw parameters: {params}")
                        
                        try:
                            # Find the matching tool to get its input schema
                            tool = next((t for t in tools if t.name == func_name), None)
                            if not tool:
                                print(f"DEBUG: Available tools: {[t.name for t in tools]}")
                                raise ValueError(f"Unknown tool: {func_name}")

                            print(f"DEBUG: Found tool: {tool.name}")
                            print(f"DEBUG: Tool schema: {tool.inputSchema}")

                            # Prepare arguments according to the tool's input schema
                            arguments = {}
                            schema_properties = tool.inputSchema.get('properties', {})
                            print(f"DEBUG: Schema properties: {schema_properties}")

                            for param_name, param_info in schema_properties.items():
                                if not params:  # Check if we have enough parameters
                                    raise ValueError(f"Not enough parameters provided for {func_name}")
                                    
                                value = params.pop(0)  # Get and remove the first parameter
                                param_type = param_info.get('type', 'string')
                                
                                print(f"DEBUG: Converting parameter {param_name} with value {value} to type {param_type}")
                                
                                # Convert the value to the correct type based on the schema
                                if param_type == 'integer':
                                    arguments[param_name] = int(value)
                                elif param_type == 'number':
                                    arguments[param_name] = float(value)
                                elif param_type == 'array':
                                    # Handle array input
                                    if isinstance(value, str):
                                        value = value.strip('[]').split(',')
                                    arguments[param_name] = [int(x.strip()) for x in value]
                                else:
                                    arguments[param_name] = str(value)

                            print(f"DEBUG: Final arguments: {arguments}")
                            print(f"DEBUG: Calling tool {func_name}")
                            
                            # Log the tool call before execution
                            start_time = time.time()
                            result = await session.call_tool(func_name, arguments=arguments)
                            execution_time = time.time() - start_time
                            print(f"DEBUG: Raw result: {result}")
                            
                            # Get the full result content from MCP response
                            if hasattr(result, 'content'):
                                print(f"DEBUG: Result has content attribute")
                                # Handle multiple content items
                                if isinstance(result.content, list):
                                    iteration_result = []
                                    for item in result.content:
                                        if hasattr(item, 'text'):
                                            iteration_result.append(item.text)
                                        elif hasattr(item, 'data'):
                                            iteration_result.append(item.data)
                                        else:
                                            iteration_result.append(str(item))
                                else:
                                    iteration_result = str(result.content)
                            else:
                                print(f"DEBUG: Result has no content attribute")
                                iteration_result = str(result)
                                
                            print(f"DEBUG: Final iteration result: {iteration_result}")
                            
                            # Log the successful tool call
                            log_tool_call(
                                tool_name=func_name,
                                arguments=arguments,
                                result=iteration_result,
                                execution_time=execution_time,
                                success=True
                            )
                            
                            # Format the response based on result type
                            if isinstance(iteration_result, list):
                                result_str = f"[{', '.join(iteration_result)}]"
                            else:
                                result_str = str(iteration_result)
                            
                            iteration_response.append(
                                f"In the {iteration + 1} iteration you called {func_name} with {arguments} parameters, "
                                f"and the function returned {result_str}."
                            )
                            last_response = iteration_result

                        except Exception as e:
                            execution_time = time.time() - start_time if 'start_time' in locals() else 0.0
                            print(f"DEBUG: Error details: {str(e)}")
                            print(f"DEBUG: Error type: {type(e)}")
                            import traceback
                            traceback.print_exc()
                            
                            # Log the failed tool call
                            log_tool_call(
                                tool_name=func_name if 'func_name' in locals() else "unknown",
                                arguments=arguments if 'arguments' in locals() else {},
                                result=None,
                                execution_time=execution_time,
                                success=False,
                                error_message=str(e)
                            )
                            
                            iteration_response.append(f"Error in iteration {iteration + 1}: {str(e)}")
                            break

                    elif response_text.startswith("FINAL_ANSWER:"):
                        print("\n=== Agent Execution Complete ===")
                        
                        # Log the final answer
                        log_tool_call(
                            tool_name="final_answer",
                            arguments={"answer": response_text},
                            result=response_text,
                            execution_time=0.0,
                            success=True
                        )
                        
                        # Log paint operations
                        start_time = time.time()
                        result = await session.call_tool("open_paint")
                        execution_time = time.time() - start_time
                        print(result.content[0].text)
                        
                        # Extract actual result content
                        if hasattr(result, 'content') and isinstance(result.content, list) and len(result.content) > 0:
                            actual_result = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                        else:
                            actual_result = str(result)
                            
                        log_tool_call(
                            tool_name="open_paint",
                            arguments={},
                            result=actual_result,
                            execution_time=execution_time,
                            success=True
                        )

                        # Wait longer for Paint to be fully maximized
                        await asyncio.sleep(1)

                        # Draw a rectangle (900x300 pt canvas)
                        start_time = time.time()
                        result = await session.call_tool(
                            "draw_rectangle",
                            arguments={
                                "x1": 50,
                                "y1": 50,
                                "x2": 850,
                                "y2": 250
                            }
                        )
                        execution_time = time.time() - start_time
                        print(result.content[0].text)
                        
                        # Extract actual result content
                        if hasattr(result, 'content') and isinstance(result.content, list) and len(result.content) > 0:
                            actual_result = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                        else:
                            actual_result = str(result)
                            
                        log_tool_call(
                            tool_name="draw_rectangle",
                            arguments={"x1": 50, "y1": 50, "x2": 850, "y2": 250},
                            result=actual_result,
                            execution_time=execution_time,
                            success=True
                        )

                        # Draw rectangle and add text
                        start_time = time.time()
                        result = await session.call_tool(
                            "add_text_in_paint",
                            arguments={
                                "text": response_text
                            }
                        )
                        execution_time = time.time() - start_time
                        print(result.content[0].text)
                        
                        # Extract actual result content
                        if hasattr(result, 'content') and isinstance(result.content, list) and len(result.content) > 0:
                            actual_result = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                        else:
                            actual_result = str(result)
                            
                        log_tool_call(
                            tool_name="add_text_in_paint",
                            arguments={"text": response_text},
                            result=actual_result,
                            execution_time=execution_time,
                            success=True
                        )
                        
                        # Keep the window open
                        start_time = time.time()
                        result = await session.call_tool("keep_window_open")
                        execution_time = time.time() - start_time
                        print(result.content[0].text)
                        
                        # Extract actual result content
                        if hasattr(result, 'content') and isinstance(result.content, list) and len(result.content) > 0:
                            actual_result = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                        else:
                            actual_result = str(result)
                            
                        log_tool_call(
                            tool_name="keep_window_open",
                            arguments={},
                            result=actual_result,
                            execution_time=execution_time,
                            success=True
                        )
                        
                        # Send email with the final result
                        start_time = time.time()
                        result = await session.call_tool(
                            "send_result_email",
                            arguments={
                                "final_answer": response_text
                            }
                        )
                        execution_time = time.time() - start_time
                        print(result.content[0].text)
                        
                        # Extract actual result content
                        if hasattr(result, 'content') and isinstance(result.content, list) and len(result.content) > 0:
                            actual_result = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                        else:
                            actual_result = str(result)
                            
                        log_tool_call(
                            tool_name="send_result_email",
                            arguments={"final_answer": response_text},
                            result=actual_result,
                            execution_time=execution_time,
                            success=True
                        )
                        break

                    iteration += 1

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # End the session and create summary
        session_summary = end_session(final_answer=last_response, query=query)
        
        # Print summary
        print("\n" + "="*50)
        print("LLM TOOL CALL SUMMARY")
        print("="*50)
        summary = get_summary()
        print(f"Session ID: {summary['session_id']}")
        print(f"Total Tool Calls: {summary['total_tool_calls']}")
        print(f"Successful Calls: {summary['successful_calls']}")
        print(f"Failed Calls: {summary['failed_calls']}")
        print(f"Total Execution Time: {summary['total_execution_time']:.2f} seconds")
        print(f"Average Execution Time: {summary['average_execution_time']:.2f} seconds")
        print("\nTool Call Counts:")
        for tool_name, count in summary['tool_call_counts'].items():
            print(f"  {tool_name}: {count}")
        
        print(f"\nDetailed log saved to: demo_llm_tool_calls.log")
        
        reset_state()  # Reset at the end of main

if __name__ == "__main__":
    asyncio.run(main())
    
    
