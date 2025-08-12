#!/usr/bin/env python3
"""
ARIana Intussusception Simulator Launcher
Simple launcher script for the medical device trainer
"""

import sys
import os
sys.dont_write_bytecode = True

# Add necessary imports for manometer integration
import serial
import threading
import queue

# This script is primarily for launching the main application
# when packaged. Dependency checks and installations are removed
# as they should be handled during the packaging process.

def main():
    """Main launcher function"""
    print("ARIana Intussusception Simulator")
    print("=" * 40)
    
    # Ensure we're in the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("Starting application...")
    
    # Import and run the application directly
    # When packaged by Nuitka, intussusception_trainer will be part of the executable
    try:
        from intussusception_trainer import ARIanaApp
        app = ARIanaApp()
        app.run()
        return 0
    except Exception as e:
        print(f"Error starting application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())


