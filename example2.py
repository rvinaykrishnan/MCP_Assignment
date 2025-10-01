# basic import 
from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent
from mcp import types
from PIL import Image as PILImage
import math
import sys
import time
import subprocess
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# instantiate an MCP server client
mcp = FastMCP("Calculator")

# Global variable to store the matplotlib figure
matplotlib_fig = None

# Load email configuration
load_dotenv()

# DEFINE TOOLS

#addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    print("CALLED: add(a: int, b: int) -> int:")
    return int(a + b)

@mcp.tool()
def add_list(l: list) -> int:
    """Add all numbers in a list"""
    print("CALLED: add(l: list) -> int:")
    return sum(l)

# subtraction tool
@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    print("CALLED: subtract(a: int, b: int) -> int:")
    return int(a - b)

# multiplication tool
@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    print("CALLED: multiply(a: int, b: int) -> int:")
    return int(a * b)

#  division tool
@mcp.tool() 
def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    print("CALLED: divide(a: int, b: int) -> float:")
    return float(a / b)

# power tool
@mcp.tool()
def power(a: int, b: int) -> int:
    """Power of two numbers"""
    print("CALLED: power(a: int, b: int) -> int:")
    return int(a ** b)

# square root tool
@mcp.tool()
def sqrt(a: int) -> float:
    """Square root of a number"""
    print("CALLED: sqrt(a: int) -> float:")
    return float(a ** 0.5)

# cube root tool
@mcp.tool()
def cbrt(a: int) -> float:
    """Cube root of a number"""
    print("CALLED: cbrt(a: int) -> float:")
    return float(a ** (1/3))

# factorial tool
@mcp.tool()
def factorial(a: int) -> int:
    """factorial of a number"""
    print("CALLED: factorial(a: int) -> int:")
    return int(math.factorial(a))

# log tool
@mcp.tool()
def log(a: int) -> float:
    """log of a number"""
    print("CALLED: log(a: int) -> float:")
    return float(math.log(a))

# remainder tool
@mcp.tool()
def remainder(a: int, b: int) -> int:
    """remainder of two numbers divison"""
    print("CALLED: remainder(a: int, b: int) -> int:")
    return int(a % b)

# sin tool
@mcp.tool()
def sin(a: int) -> float:
    """sin of a number"""
    print("CALLED: sin(a: int) -> float:")
    return float(math.sin(a))

# cos tool
@mcp.tool()
def cos(a: int) -> float:
    """cos of a number"""
    print("CALLED: cos(a: int) -> float:")
    return float(math.cos(a))

# tan tool
@mcp.tool()
def tan(a: int) -> float:
    """tan of a number"""
    print("CALLED: tan(a: int) -> float:")
    return float(math.tan(a))

# mine tool
@mcp.tool()
def mine(a: int, b: int) -> int:
    """special mining tool"""
    print("CALLED: mine(a: int, b: int) -> int:")
    return int(a - b - b)

@mcp.tool()
def create_thumbnail(image_path: str) -> Image:
    """Create a thumbnail from an image"""
    print("CALLED: create_thumbnail(image_path: str) -> Image:")
    img = PILImage.open(image_path)
    img.thumbnail((100, 100))
    return Image(data=img.tobytes(), format="png")

@mcp.tool()
def strings_to_chars_to_int(string: str) -> list[int]:
    """Return the ASCII values of the characters in a word"""
    print("CALLED: strings_to_chars_to_int(string: str) -> list[int]:")
    return [int(ord(char)) for char in string]

@mcp.tool()
def int_list_to_exponential_sum(int_list: list) -> float:
    """Return sum of exponentials of numbers in a list"""
    print("CALLED: int_list_to_exponential_sum(int_list: list) -> float:")
    return sum(math.exp(i) for i in int_list)

@mcp.tool()
def fibonacci_numbers(n: int) -> list:
    """Return the first n Fibonacci Numbers"""
    print("CALLED: fibonacci_numbers(n: int) -> list:")
    if n <= 0:
        return []
    fib_sequence = [0, 1]
    for _ in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence[:n]


@mcp.tool()
async def draw_rectangle(x1: int, y1: int, x2: int, y2: int) -> dict:
    """Draw a rectangle using matplotlib from (x1,y1) to (x2,y2)"""
    global matplotlib_fig
    try:
        if matplotlib_fig is None:
            return {
                "content": [
                    TextContent(
                        type="text",
                        text="Matplotlib figure is not open. Please call open_paint first."
                    )
                ]
            }
        
        # Get the current axes
        ax = matplotlib_fig.axes[0]
        
        # Calculate rectangle dimensions
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        left = min(x1, x2)
        bottom = min(y1, y2)
        
        # Create rectangle patch
        rectangle = patches.Rectangle((left, bottom), width, height, 
                                    linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.3)
        ax.add_patch(rectangle)
        
        # Refresh the display and keep window open
        matplotlib_fig.canvas.draw()
        matplotlib_fig.canvas.flush_events()
        plt.pause(0.1)
        
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"Rectangle drawn from ({x1},{y1}) to ({x2},{y2}) with width {width} and height {height}"
                )
            ]
        }
    except Exception as e:
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"Error drawing rectangle: {str(e)}"
                )
            ]
        }

@mcp.tool()
async def add_text_in_paint(text: str) -> dict:
    """Add text in the matplotlib figure"""
    global matplotlib_fig
    try:
        if matplotlib_fig is None:
            return {
                "content": [
                    TextContent(
                        type="text",
                        text="Matplotlib figure is not open. Please call open_paint first."
                    )
                ]
            }
        
        # Get the current axes
        ax = matplotlib_fig.axes[0]
        
        # Format the text to show "Final Answer = [value]"
        if text.startswith("FINAL_ANSWER:"):
            # Extract the value from "FINAL_ANSWER: [value]"
            value = text.replace("FINAL_ANSWER:", "").strip()
            formatted_text = f"Final Answer = {value}"
        else:
            formatted_text = text
        
        # Add text in the center of the canvas (450, 150) with font size 20
        ax.text(450, 150, formatted_text, fontsize=20, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                fontweight='bold')
        
        # Refresh the display and keep window open
        matplotlib_fig.canvas.draw()
        matplotlib_fig.canvas.flush_events()
        plt.pause(0.1)
        
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"Text '{text}' added successfully in the center with font size 20"
                )
            ]
        }
    except Exception as e:
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"Error adding text: {str(e)}"
                )
            ]
        }

@mcp.tool()
async def open_paint() -> dict:
    """Create a matplotlib figure for drawing"""
    global matplotlib_fig
    try:
        # Create a new matplotlib figure with specified dimensions
        matplotlib_fig, ax = plt.subplots(1, 1, figsize=(12, 4))  # 12x4 inches for 900x300 pt equivalent
        ax.set_xlim(0, 900)
        ax.set_ylim(0, 300)
        ax.set_aspect('equal')
        ax.set_facecolor('white')
        ax.axis('off')  # Hide axes
        
        # Set the figure title
        matplotlib_fig.suptitle('Mathematical Results Display', fontsize=16, fontweight='bold')
        
        # Show the figure and keep it open
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to ensure the window opens
        
        # Keep the figure alive
        matplotlib_fig.canvas.draw_idle()
        
        return {
            "content": [
                TextContent(
                    type="text",
                    text="Matplotlib figure created successfully (900x300 pt canvas)"
                )
            ]
        }
    except Exception as e:
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"Error creating matplotlib figure: {str(e)}"
                )
            ]
        }
# DEFINE RESOURCES

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    print("CALLED: get_greeting(name: str) -> str:")
    return f"Hello, {name}!"


# DEFINE AVAILABLE PROMPTS
@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"
    print("CALLED: review_code(code: str) -> str:")


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]

@mcp.tool()
async def keep_window_open() -> dict:
    """Keep the matplotlib window open indefinitely"""
    global matplotlib_fig
    try:
        if matplotlib_fig is None:
            return {
                "content": [
                    TextContent(
                        type="text",
                        text="No matplotlib figure is open. Please call open_paint first."
                    )
                ]
            }
        
        # Keep the window open by showing it with block=True
        plt.show(block=True)
        
        return {
            "content": [
                TextContent(
                    type="text",
                    text="Matplotlib window is now open and will remain visible until manually closed"
                )
            ]
        }
    except Exception as e:
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"Error keeping window open: {str(e)}"
                )
            ]
        }

@mcp.tool()
async def send_result_email(final_answer: str) -> dict:
    """Send the final result via email"""
    try:
        # Get email configuration
        gmail_address = os.getenv('GMAIL_ADDRESS')
        gmail_password = os.getenv('GMAIL_PASSWORD')
        recipient_email = os.getenv('RECIPIENT_EMAIL')
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        
        if not all([gmail_address, gmail_password, recipient_email]):
            return {
                "content": [
                    TextContent(
                        type="text",
                        text="Email configuration missing. Please check .env file."
                    )
                ]
            }
        
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = gmail_address
        msg['To'] = recipient_email
        msg['Subject'] = "Mathematical Calculation Result"
        
        # Email body
        body = f"""
        Mathematical Calculation Result
        
        The calculation has been completed successfully.
        
        Final Answer: {final_answer}
        
        This result was generated by the MCP Mathematical Calculator.
        
        Best regards,
        MCP Calculator System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Enable TLS encryption
        server.login(gmail_address, gmail_password)
        text = msg.as_string()
        server.sendmail(gmail_address, recipient_email, text)
        server.quit()
        
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"Email sent successfully to {recipient_email} with result: {final_answer}"
                )
            ]
        }
        
    except Exception as e:
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"Error sending email: {str(e)}"
                )
            ]
        }

if __name__ == "__main__":
    # Check if running with mcp dev command
    print("STARTING THE SERVER AT AMAZING LOCATION")
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        mcp.run()  # Run without transport for dev server
    else:
        mcp.run(transport="stdio")  # Run with stdio for direct execution
